#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import math
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "src" / "analysis"
SIMULATIONS_DIR = REPO_ROOT / "simulations"
for import_root in (str(REPO_ROOT), str(ANALYSIS_DIR), str(SIMULATIONS_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from adaptive_methods import (  # type: ignore  # noqa: E402
    build_common_args,
    build_grid,
    load_json,
    nearest_grid_value,
    read_csv_rows,
    run_checked,
    run_eq_window as run_eq_window_raw,
    run_neq_edge,
    write_csv as write_csv_raw,
    write_json as write_json_raw,
    write_protocol_path as write_protocol_path_raw,
)
from bidirectional_mts_pmf import (  # type: ignore  # noqa: E402
    bootstrap_bidirectional_mts_pmf,
    solve_segment_cft_delta_f_once,
    trajectory_frames_to_arrays,
)
from mines_current_protocol_analysis import (  # type: ignore  # noqa: E402
    bootstrap_direct_eq_mbar,
    direct_eq_mbar_pmf,
    pair_js_divergence,
)
from mines_notebook_utils import (  # type: ignore  # noqa: E402
    background_potential_1d,
    coverage_mask_from_samples,
    mode_x_from_samples,
)


@contextmanager
def timed_operation(
    timing_rows: list[dict[str, Any]],
    *,
    stage: str,
    operation: str,
    item: str = "",
    metadata: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
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


@dataclass
class EnsembleWindow:
    name: str
    center_x: float
    k: float
    root: Path
    eq_file: Path
    tail_file: Path
    eq_rows: list[dict[str, str]]
    tail_rows: list[dict[str, str]]
    mean_x: float
    std_x: float
    x_most: float
    generation: int
    side: str


@dataclass
class EQCluster:
    name: str
    windows: list[EnsembleWindow]
    left_x: float
    right_x: float


@dataclass
class NEQSegment:
    name: str
    left: EQCluster | EnsembleWindow
    right: EQCluster | EnsembleWindow
    left_boundary: EnsembleWindow
    right_boundary: EnsembleWindow
    root: Path
    forward_trajectories: list[list[dict[str, str]]]
    reverse_trajectories: list[list[dict[str, str]]]
    forward_trajectory_files: list[Path]
    reverse_trajectory_files: list[Path]
    forward_path_file: Path
    reverse_path_file: Path
    protocol_k: float
    n_neq_traj_requested: int = 0
    n_neq_traj_actual: int = 0
    neq_budget_limited: bool = False
    neq_cost_requested: int = 0
    neq_cost_actual: int = 0
    remaining_budget_before_segment: int = 0
    protocol_mode: str = "GT"
    protocol_metadata: dict[str, Any] = field(default_factory=dict)
    protocol_k_min: float | None = None
    protocol_k_max: float | None = None
    protocol_x_min: float | None = None
    protocol_x_max: float | None = None
    protocol_clip_fraction_k: float = 0.0
    protocol_clip_fraction_x: float = 0.0
    connectivity: dict[str, Any] = field(default_factory=dict)
    mts_patch_built: bool = False
    cft_summary: dict[str, Any] = field(default_factory=dict)
    neq_patch_decision: dict[str, Any] = field(default_factory=dict)
    hs_fallback_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PMFPatch:
    name: str
    kind: str
    root: Path
    grid: np.ndarray
    pmf: np.ndarray
    variance: np.ndarray
    coverage_mask: np.ndarray
    source_names: list[str]
    metadata: dict[str, Any]
    anchor_variances: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class BudgetTracker:
    total_budget_steps: int
    used_steps: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def can_spend(self, cost: int) -> bool:
        return int(self.used_steps) + int(cost) <= int(self.total_budget_steps)

    def spend(self, cost: int, label: str, kind: str, stage: str) -> None:
        if not self.can_spend(int(cost)):
            raise RuntimeError(
                f"Budget exceeded before {kind} {label} at {stage}: "
                f"need {int(cost)} more steps, used {int(self.used_steps)} of "
                f"{int(self.total_budget_steps)}."
            )
        self.used_steps += int(cost)
        self.ledger.append(
            {
                "stage": str(stage),
                "item": str(label),
                "kind": str(kind),
                "cost": int(cost),
                "cumulative_used": int(self.used_steps),
                "budget": int(self.total_budget_steps),
            }
        )

    def write(self, path: Path) -> None:
        write_csv(
            path,
            ["stage", "item", "kind", "cost", "cumulative_used", "budget"],
            self.ledger,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean MiNES workflow with variance-weighted PMF fusion.",
    )
    parser.add_argument("--system-root", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--label", default="mines_variance_fusion")
    parser.add_argument("--total-budget-steps", default=2500000, type=int)
    parser.add_argument("--t-neq", default=5000, type=int)
    parser.add_argument("--n-neq-traj", default=100, type=int)
    parser.add_argument("--n-eq-steps", default=10000, type=int)
    parser.add_argument("--eq-save-every", default=10, type=int)
    parser.add_argument("--tail-fraction", default=0.9, type=float)
    parser.add_argument("--q-next", default=0.9, type=float)
    parser.add_argument("--alpha", default=2.0, type=float)
    parser.add_argument("--x-leap", default=1.5, type=float)
    parser.add_argument("--k-min", default=1.0, type=float)
    parser.add_argument("--k-max", default=100.0, type=float)
    parser.add_argument("--k-rescue", default=10.0, type=float)
    parser.add_argument("--js-threshold", default=0.3, type=float)
    parser.add_argument("--neq-connectivity-threshold", default=0.3, type=float)
    parser.add_argument("--bin-width", default=0.1, type=float)
    parser.add_argument("--max-generations", default=10, type=int)
    parser.add_argument(
        "--max-rescue-rounds",
        default=None,
        type=int,
        help="Deprecated. Kept only for backward compatibility; final EQ refinement is budget-limited instead.",
    )
    parser.add_argument(
        "--final-refinement-mode",
        choices=["none", "eq-extend"],
        default="eq-extend",
        help="Final refinement after all EQ ensembles are connected. eq-extend extends existing EQ windows until target MBAR ddF or budget exhaustion.",
    )
    parser.add_argument(
        "--target-mbar-ddf",
        default=1.0e-3,
        type=float,
        help="Target maximum connected-EQ MBAR uncertainty (sqrt(variance)) for final EQ-extension refinement.",
    )
    parser.add_argument(
        "--eq-extension-steps",
        default=None,
        type=int,
        help="Additional EQ steps per selected window per final EQ-extension round. If None, uses --n-eq-steps.",
    )
    parser.add_argument("--n-bootstrap-eq", default=64, type=int)
    parser.add_argument("--n-bootstrap-neq", default=64, type=int)
    parser.add_argument("--variance-floor", default=1.0e-6, type=float)
    parser.add_argument("--allow-partial-neq-budget", action="store_true")
    parser.add_argument("--min-neq-traj", default=5, type=int)
    parser.add_argument("--analysis-xmin", type=float, default=None)
    parser.add_argument("--analysis-xmax", type=float, default=None)
    parser.add_argument("--disable-fixed-cft-bootstrap", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--s-rescue", default=2.0, type=float)
    parser.add_argument("--rescue-center-f-slope", default=0.5, type=float)
    parser.add_argument("--rescue-center-f-start", default=2.0, type=float)
    parser.add_argument("--rescue-center-f-min", default=-2.0, type=float)
    parser.add_argument("--rescue-center-f-max", default=1.0, type=float)
    parser.add_argument(
        "--neq-protocol-mode",
        choices=["GT", "linear", "gt"],
        default="GT",
        help="NEQ bridge protocol mode. GT uses local Gaussian transport; linear interpolates x and sqrt(k).",
    )
    parser.add_argument(
        "--pmf-method",
        choices=["neq", "eq", "hybrid"],
        default="neq",
        help="Which patches to use for the global PMF fit: neq=only NEQ/MTS, hybrid=EQ+NEQ, eq=EQ pmf+NEQ variance.",
    )
    parser.add_argument(
        "--cft-ddf-threshold",
        default=1.0,
        type=float,
        help="Stop chain growth when CFT delta_f falls below this threshold.",
    )
    parser.add_argument(
        "--rescue-background-fit-method",
        choices=["global-pmf", "mean-only", "auto"],
        default="global-pmf",
        help="Background fit method for rescue window design. global-pmf=fit global fused PMF locally with mean-only fallback; mean-only=mean-only GT only; auto=same as global-pmf with explicit fallback diagnostics.",
    )
    parser.add_argument("--rescue-global-fit-min-bins", default=5, type=int,
        help="Minimum number of valid bins for global-PMF quadratic fit acceptance.")
    parser.add_argument("--rescue-global-fit-k0-min-abs", default=1.0e-8, type=float,
        help="Minimum |k0| for global-PMF quadratic fit acceptance.")
    parser.add_argument("--rescue-global-fit-x0-margin-factor", default=0.25, type=float,
        help="Allowed x0 margin as a fraction of segment width beyond boundary means.")
    parser.add_argument("--rescue-global-fit-radius", default=None, type=float,
        help="If set, restrict global-PMF fit domain to ±radius around target_bin_x.")
    # Deprecated aliases kept for backward compat
    parser.add_argument("--rescue-neq-fit-min-bins", default=5, type=int,
        help="Deprecated alias for --rescue-global-fit-min-bins.")
    parser.add_argument("--rescue-neq-fit-k0-min-abs", default=1.0e-8, type=float,
        help="Deprecated alias for --rescue-global-fit-k0-min-abs.")
    parser.add_argument("--rescue-neq-fit-x0-margin-factor", default=0.25, type=float,
        help="Deprecated alias for --rescue-global-fit-x0-margin-factor.")
    return parser.parse_args()


def apply_quick_test_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(args.quick_test):
        return args
    args.n_eq_steps = 1000
    args.t_neq = 200
    args.n_neq_traj = 10
    args.max_generations = 1
    args.final_refinement_mode = "none"
    args.n_bootstrap_eq = 8
    args.n_bootstrap_neq = 8
    args.total_budget_steps = 20000
    args.js_threshold = 0.0
    return args


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_raw(path, json_ready(payload))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    write_csv_raw(path, fieldnames, [json_ready(row) for row in rows])


def write_protocol_path(path: Path, centers: list[float], ks: list[float]) -> None:
    write_protocol_path_raw(path, centers, ks)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def compute_pmf_quality_metrics(
    *,
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: np.ndarray,
    ctx: dict[str, Any],
    used_steps: int,
    stage: str,
    analysis_xmin: float,
    analysis_xmax: float,
) -> dict[str, Any]:
    grid = np.asarray(grid, dtype=float)
    global_pmf = np.asarray(global_pmf, dtype=float)
    global_variance = np.asarray(global_variance, dtype=float)
    analytic = analytic_pmf(grid, ctx)
    half_dx = 0.5 * abs(float(grid[1] - grid[0])) if len(grid) > 1 else 0.05
    analysis_mask = (
        np.isfinite(grid)
        & (grid >= float(analysis_xmin) - half_dx)
        & (grid <= float(analysis_xmax) + half_dx)
    )
    n_interest_bins = int(np.count_nonzero(analysis_mask))
    covered_mask = analysis_mask & np.isfinite(global_pmf)
    n_covered_bins = int(np.count_nonzero(covered_mask))
    uncovered_mask = analysis_mask & ~np.isfinite(global_pmf)
    n_uncovered_bins = int(np.count_nonzero(uncovered_mask))
    uncovered_xs = grid[uncovered_mask]
    if len(uncovered_xs) > 0:
        first_uncovered_x = float(uncovered_xs[0])
        last_uncovered_x = float(uncovered_xs[-1])
        _ux_strs = [f"{v:.4g}" for v in uncovered_xs[:50]]
        uncovered_x_values = ";".join(_ux_strs) + (
            f";...({len(uncovered_xs) - 50} more)" if len(uncovered_xs) > 50 else ""
        )
    else:
        first_uncovered_x = float("nan")
        last_uncovered_x = float("nan")
        uncovered_x_values = ""
    coverage_fraction = (
        float(n_covered_bins) / float(n_interest_bins)
        if n_interest_bins > 0
        else float("nan")
    )
    coverage_percent = (
        100.0 * float(coverage_fraction)
        if math.isfinite(float(coverage_fraction))
        else float("nan")
    )
    error_mask = analysis_mask & np.isfinite(global_pmf) & np.isfinite(analytic)
    n_error_bins = int(np.count_nonzero(error_mask))
    if n_error_bins > 0:
        bestfit_offset = float(np.nanmean(analytic[error_mask] - global_pmf[error_mask]))
        rmse_bestfit = float(
            np.sqrt(
                np.nanmean(
                    np.square((global_pmf[error_mask] + bestfit_offset) - analytic[error_mask])
                )
            )
        )
    else:
        bestfit_offset = float("nan")
        rmse_bestfit = float("nan")
    variance_mask = analysis_mask & np.isfinite(global_pmf) & np.isfinite(global_variance)
    mean_global_variance = (
        float(np.nanmean(global_variance[variance_mask]))
        if np.any(variance_mask)
        else float("nan")
    )
    median_global_variance = (
        float(np.nanmedian(global_variance[variance_mask]))
        if np.any(variance_mask)
        else float("nan")
    )
    if np.any(variance_mask):
        max_idx = int(np.argmax(global_variance[variance_mask]))
        _var_vals = global_variance[variance_mask]
        _var_xs = grid[variance_mask]
        max_global_variance = float(_var_vals[max_idx])
        x_at_max_global_variance = float(_var_xs[max_idx])
        max_global_std = float(math.sqrt(max_global_variance)) if max_global_variance >= 0.0 else float("nan")
    else:
        max_global_variance = float("nan")
        x_at_max_global_variance = float("nan")
        max_global_std = float("nan")
    return {
        "stage": str(stage),
        "used_steps": int(used_steps),
        "used_ksteps": float(used_steps) / 1000.0,
        "analysis_xmin": float(analysis_xmin),
        "analysis_xmax": float(analysis_xmax),
        "n_interest_bins": int(n_interest_bins),
        "n_covered_bins": int(n_covered_bins),
        "n_uncovered_bins": int(n_uncovered_bins),
        "first_uncovered_x": float(first_uncovered_x),
        "last_uncovered_x": float(last_uncovered_x),
        "uncovered_x_values": str(uncovered_x_values),
        "coverage_fraction": float(coverage_fraction),
        "coverage_percent": float(coverage_percent),
        "rmse_bestfit": float(rmse_bestfit),
        "bestfit_offset": float(bestfit_offset),
        "n_error_bins": int(n_error_bins),
        "mean_global_variance": float(mean_global_variance),
        "median_global_variance": float(median_global_variance),
        "max_global_variance": float(max_global_variance),
        "x_at_max_global_variance": float(x_at_max_global_variance),
        "max_global_std": float(max_global_std),
    }


def analytic_pmf(grid: np.ndarray, ctx: dict[str, Any]) -> np.ndarray:
    return np.asarray(background_potential_1d(grid, ctx), dtype=float)


def choose_affordable_neq_count(
    *,
    requested_n_neq: int,
    t_neq: int,
    budget: BudgetTracker,
    allow_partial: bool,
    min_neq_traj: int,
) -> dict[str, Any]:
    requested_n_neq = int(requested_n_neq)
    t_neq = int(t_neq)
    min_neq_traj = int(min_neq_traj)
    cost_requested = stage_cost_neq(requested_n_neq, t_neq)
    remaining = int(budget.total_budget_steps) - int(budget.used_steps)
    if budget.can_spend(cost_requested):
        return {
            "can_run": True,
            "n_neq_traj_requested": requested_n_neq,
            "n_neq_traj_actual": requested_n_neq,
            "neq_budget_limited": False,
            "cost_requested": int(cost_requested),
            "cost_actual": int(cost_requested),
            "remaining_budget_before_segment": int(remaining),
            "min_neq_traj": min_neq_traj,
            "reason": "full_neq_fits_budget",
        }
    if bool(allow_partial):
        affordable_n = int(max(0, remaining // max(2 * t_neq, 1)))
        if affordable_n >= min_neq_traj:
            cost_actual = stage_cost_neq(affordable_n, t_neq)
            return {
                "can_run": True,
                "n_neq_traj_requested": requested_n_neq,
                "n_neq_traj_actual": affordable_n,
                "neq_budget_limited": True,
                "cost_requested": int(cost_requested),
                "cost_actual": int(cost_actual),
                "remaining_budget_before_segment": int(remaining),
                "min_neq_traj": min_neq_traj,
                "reason": "partial_neq_budget_used",
            }
        return {
            "can_run": False,
            "n_neq_traj_requested": requested_n_neq,
            "n_neq_traj_actual": 0,
            "neq_budget_limited": True,
            "cost_requested": int(cost_requested),
            "cost_actual": 0,
            "remaining_budget_before_segment": int(remaining),
            "min_neq_traj": min_neq_traj,
            "reason": "remaining_budget_below_min_neq_traj",
        }
    return {
        "can_run": False,
        "n_neq_traj_requested": requested_n_neq,
        "n_neq_traj_actual": 0,
        "neq_budget_limited": False,
        "cost_requested": int(cost_requested),
        "cost_actual": 0,
        "remaining_budget_before_segment": int(remaining),
        "min_neq_traj": min_neq_traj,
        "reason": "partial_neq_budget_disabled",
    }


def normalized_jsd(jsd_raw: float) -> float:
    if not math.isfinite(float(jsd_raw)):
        return float("nan")
    return float(jsd_raw) / math.log(2.0)


def normalized_jsd_clipped(jsd_raw: float) -> float:
    value = normalized_jsd(jsd_raw)
    if not math.isfinite(value):
        return value
    return float(min(max(value, 0.0), 1.0))


def logsumexp_np(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if axis is None:
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.asarray(-np.inf, dtype=float)
        max_value = np.nanmax(arr[finite])
        return np.asarray(max_value + np.log(np.sum(np.exp(arr[finite] - max_value))), dtype=float)
    max_value = np.nanmax(arr, axis=axis, keepdims=True)
    finite_max = np.isfinite(max_value)
    safe_max = np.where(finite_max, max_value, 0.0)
    shifted = np.exp(arr - safe_max)
    shifted = np.where(np.isfinite(arr), shifted, 0.0)
    summed = np.sum(shifted, axis=axis, keepdims=True)
    result = safe_max + np.log(summed)
    result = np.where(finite_max, result, -np.inf)
    return np.squeeze(result, axis=axis)


def window_dirname(name: str) -> str:
    return str(name).lower()


def shift_finite_to_min_zero(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(shifted)
    if np.any(finite):
        shifted[finite] -= float(np.nanmin(shifted[finite]))
    return shifted


def ordered_fieldnames(rows: list[dict[str, Any]], extras: list[str] | None = None) -> list[str]:
    if not rows:
        return list(extras or [])
    names = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in names:
                names.append(key)
    for key in extras or []:
        if key not in names:
            names.append(key)
    return names


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def normalize_flat_neq_outputs_to_segment_layout(segment_root: Path) -> None:
    canonical_forward = sorted(segment_root.glob("forward/traj_*/neq_fwd_0.csv"))
    canonical_reverse = sorted(segment_root.glob("reverse/traj_*/neq_fwd_0.csv"))
    if canonical_forward and canonical_reverse:
        return
    flat_forward = sorted(segment_root.glob("neq_fwd_*.csv"))
    flat_reverse_bwd = sorted(segment_root.glob("neq_bwd_*.csv"))
    flat_reverse_rev = sorted(segment_root.glob("neq_rev_*.csv"))
    reverse_flat = flat_reverse_bwd if flat_reverse_bwd else flat_reverse_rev
    if not flat_forward or not reverse_flat:
        return
    if len(flat_forward) != len(reverse_flat):
        raise RuntimeError(
            f"Mismatched flat NEQ output counts in {segment_root}: "
            f"{len(flat_forward)} forward vs {len(reverse_flat)} reverse."
        )
    for idx, path in enumerate(flat_forward):
        dst = segment_root / "forward" / f"traj_{idx:03d}" / "neq_fwd_0.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dst))
    for idx, path in enumerate(reverse_flat):
        dst = segment_root / "reverse" / f"traj_{idx:03d}" / "neq_fwd_0.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dst))


def discover_neq_trajectories(segment_root: Path) -> tuple[list[Path], list[Path]]:
    canonical_forward = sorted(segment_root.glob("forward/traj_*/neq_fwd_0.csv"))
    canonical_reverse = sorted(segment_root.glob("reverse/traj_*/neq_fwd_0.csv"))
    if canonical_forward and canonical_reverse:
        return canonical_forward, canonical_reverse
    flat_forward = sorted(segment_root.glob("neq_fwd_*.csv"))
    flat_reverse_bwd = sorted(segment_root.glob("neq_bwd_*.csv"))
    if flat_forward and flat_reverse_bwd:
        return flat_forward, flat_reverse_bwd
    flat_reverse_rev = sorted(segment_root.glob("neq_rev_*.csv"))
    if flat_forward and flat_reverse_rev:
        return flat_forward, flat_reverse_rev
    raise RuntimeError(f"Could not discover bidirectional NEQ trajectories in {segment_root}.")


def summarize_trajectory_rows(
    traj_paths: list[Path],
    traj_rows_list: list[list[dict[str, str]]],
    out_dir: Path,
    endpoint_csv_name: str,
    relative_root: Path,
) -> tuple[Path, Path]:
    drawn_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for idx, (traj_path, rows) in enumerate(zip(traj_paths, traj_rows_list)):
        if not rows:
            continue
        first_row = rows[0]
        last_row = rows[-1]
        drawn_rows.append(
            {
                "draw_idx": int(idx),
                "start_step": int(float(first_row.get("step", "0") or 0.0)),
                "start_lambda": float(first_row.get("lambda", "0.0") or 0.0),
                "start_x": float(first_row["x"]),
                "start_y": float(first_row.get("y", "0.0") or 0.0),
                "traj_file": str(traj_path.relative_to(relative_root)),
            }
        )
        endpoint_rows.append(
            {
                "draw_idx": int(idx),
                "final_step": int(float(last_row.get("step", "0") or 0.0)),
                "final_lambda": float(last_row.get("lambda", "0.0") or 0.0),
                "final_x": float(last_row["x"]),
                "final_y": float(last_row.get("y", "0.0") or 0.0),
                "final_work": float(last_row.get("work", "nan") or "nan"),
                "traj_file": str(traj_path.relative_to(relative_root)),
            }
        )
    draws_file = out_dir / "drawn_start_samples.csv"
    endpoints_file = out_dir / endpoint_csv_name
    write_csv(
        draws_file,
        ordered_fieldnames(
            drawn_rows,
            extras=["draw_idx", "start_step", "start_lambda", "start_x", "start_y", "traj_file"],
        ),
        drawn_rows,
    )
    write_csv(
        endpoints_file,
        ordered_fieldnames(
            endpoint_rows,
            extras=["draw_idx", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
        ),
        endpoint_rows,
    )
    return draws_file, endpoints_file


def tail_rows_from_eq_rows(
    eq_rows: list[dict[str, str]],
    tail_fraction: float,
) -> list[dict[str, str]]:
    if not eq_rows:
        return []
    fraction = float(tail_fraction)
    if fraction >= 1.0:
        return list(eq_rows)
    if fraction <= 0.0:
        return [eq_rows[-1]]
    discard = int(math.floor(len(eq_rows) * (1.0 - fraction)))
    discard = min(max(discard, 0), max(len(eq_rows) - 1, 0))
    return list(eq_rows[discard:])


def eq_tail_samples(window: EnsembleWindow) -> np.ndarray:
    values = [float(row["x"]) for row in window.tail_rows if row.get("x", "") != ""]
    return np.asarray(values, dtype=float)


def compute_eq_pair_bar_summary(
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    _FAIL: dict[str, Any] = {
        "bar_solved": False,
        "bar_delta_f": "",
        "bar_delta_f_unc": "",
        "bar_method": "",
        "bar_reason": "",
    }
    try:
        x_L = eq_tail_samples(left_window)
        x_R = eq_tail_samples(right_window)
        x_L = x_L[np.isfinite(x_L)]
        x_R = x_R[np.isfinite(x_R)]
        if x_L.size < 3 or x_R.size < 3:
            return {**_FAIL, "bar_reason": "not_enough_samples"}
        kT = max(float(ctx.get("thermal_kT", 1.0)), 1e-12)
        beta = 1.0 / kT
        k_L, cx_L = float(left_window.k), float(left_window.center_x)
        k_R, cx_R = float(right_window.k), float(right_window.center_x)
        # Reduced potential difference: perturb left samples under right bias and vice versa
        w_F = beta * (0.5 * k_R * (x_L - cx_R) ** 2 - 0.5 * k_L * (x_L - cx_L) ** 2)
        w_R = beta * (0.5 * k_L * (x_R - cx_L) ** 2 - 0.5 * k_R * (x_R - cx_R) ** 2)
        if not (np.isfinite(w_F).all() and np.isfinite(w_R).all()):
            return {**_FAIL, "bar_reason": "nonfinite_work_values"}
        # Reshape to (n_samples, 1) so solve_segment_cft_delta_f_once sees 2-D arrays
        cft = solve_segment_cft_delta_f_once(
            w_F.reshape(-1, 1), w_R.reshape(-1, 1), kT=kT
        )
        if bool(cft.get("cft_solved", False)):
            return {
                "bar_solved": True,
                "bar_delta_f": cft.get("delta_f", ""),
                "bar_delta_f_unc": cft.get("delta_f_unc", ""),
                "bar_method": cft.get("method", "BAR"),
                "bar_reason": cft.get("reason", ""),
            }
        return {**_FAIL, "bar_reason": f"bar_solver_failed: {cft.get('reason', '')}"}
    except Exception as exc:
        return {**_FAIL, "bar_reason": f"bar_solver_failed: {exc}"}


def pair_jsd_metrics(left_x: np.ndarray, right_x: np.ndarray, grid: np.ndarray) -> dict[str, Any]:
    js_raw = float(pair_js_divergence(left_x, right_x, grid))
    js_norm = normalized_jsd(js_raw)
    return {
        "pair_jsd_raw": js_raw,
        "pair_jsd_norm": js_norm,
        "pair_jsd": js_norm,
    }


def eop_distributions_cross(
    x_eop_fwd: np.ndarray,
    x_eop_rev: np.ndarray,
    grid: np.ndarray,
    js_threshold_norm: float,
) -> dict[str, Any]:
    fwd = np.asarray(x_eop_fwd, dtype=float)
    rev = np.asarray(x_eop_rev, dtype=float)
    fwd = fwd[np.isfinite(fwd)]
    rev = rev[np.isfinite(rev)]
    js_metrics = pair_jsd_metrics(fwd, rev, grid) if fwd.size and rev.size else {
        "pair_jsd_raw": float("nan"),
        "pair_jsd_norm": float("nan"),
        "pair_jsd": float("nan"),
    }
    support_fwd = coverage_mask_from_samples(fwd, grid) if fwd.size else np.zeros(len(grid), dtype=bool)
    support_rev = coverage_mask_from_samples(rev, grid) if rev.size else np.zeros(len(grid), dtype=bool)
    common_bins = int(np.count_nonzero(np.asarray(support_fwd, dtype=bool) & np.asarray(support_rev, dtype=bool)))
    fwd_min = float(np.nanmin(fwd)) if fwd.size else float("nan")
    fwd_max = float(np.nanmax(fwd)) if fwd.size else float("nan")
    rev_min = float(np.nanmin(rev)) if rev.size else float("nan")
    rev_max = float(np.nanmax(rev)) if rev.size else float("nan")
    interval_overlap = (
        max(min(fwd_max, rev_max) - max(fwd_min, rev_min), 0.0)
        if np.isfinite(fwd_min) and np.isfinite(fwd_max) and np.isfinite(rev_min) and np.isfinite(rev_max)
        else float("nan")
    )
    cross_by_jsd = bool(
        math.isfinite(float(js_metrics["pair_jsd_norm"]))
        and float(js_metrics["pair_jsd_norm"]) <= float(js_threshold_norm)
    )
    cross_by_common_bins = bool(common_bins > 0)
    eop_crossed = bool(cross_by_jsd or cross_by_common_bins)
    return {
        "eop_crossed": eop_crossed,
        "eop_jsd_raw": float(js_metrics["pair_jsd_raw"]),
        "eop_jsd_norm": float(js_metrics["pair_jsd_norm"]),
        "eop_jsd_threshold": float(js_threshold_norm),
        "eop_common_bins": common_bins,
        "eop_fwd_min": fwd_min,
        "eop_fwd_max": fwd_max,
        "eop_rev_min": rev_min,
        "eop_rev_max": rev_max,
        "eop_interval_overlap": float(interval_overlap),
        "cross_by_jsd": cross_by_jsd,
        "cross_by_common_bins": cross_by_common_bins,
    }


def relative_to_root(path: Path, out_root: Path) -> str:
    try:
        return str(path.relative_to(out_root))
    except ValueError:
        return str(path)


def build_window_summary(window: EnsembleWindow, out_root: Path) -> dict[str, Any]:
    return {
        "name": window.name,
        "center_x": float(window.center_x),
        "k": float(window.k),
        "mean_x": float(window.mean_x),
        "std_x": float(window.std_x),
        "x_most": float(window.x_most),
        "mean_minus_x_most": float(window.mean_x - window.x_most),
        "mean_minus_center_x": float(window.mean_x - window.center_x),
        "x_most_minus_center_x": float(window.x_most - window.center_x),
        "eq_file": relative_to_root(window.eq_file, out_root),
        "tail_file": relative_to_root(window.tail_file, out_root),
        "n_eq_samples": int(len(window.eq_rows)),
        "n_tail_samples": int(len(window.tail_rows)),
        "generation": int(window.generation),
        "side": window.side,
    }


def run_eq_window(
    *,
    name: str,
    center_x: float,
    k: float,
    generation: int,
    side: str,
    bin_path: str,
    ctx: dict[str, Any],
    grid: np.ndarray,
    n_eq_steps: int,
    eq_save_every: int,
    tail_fraction: float,
    seed: int,
    root: Path,
    out_root: Path,
) -> EnsembleWindow:
    root.mkdir(parents=True, exist_ok=True)
    nout = max(1, int(math.ceil(float(n_eq_steps) / max(int(eq_save_every), 1))))
    run_eq_window_raw(
        bin_path=bin_path,
        ctx=ctx,
        center_x=float(center_x),
        k=float(k),
        steps=int(n_eq_steps),
        nout=int(nout),
        seed=int(seed),
        out_dir=root,
    )
    eq_file = root / "eq_window.csv"
    eq_rows = read_csv_rows(eq_file)
    if not eq_rows:
        raise RuntimeError(f"EQ run for {name} wrote no samples: {eq_file}")
    tail_rows = tail_rows_from_eq_rows(eq_rows, tail_fraction)
    tail_file = root / "eq_tail.csv"
    write_csv(tail_file, ordered_fieldnames(eq_rows), tail_rows)
    tail_x = np.asarray([float(row["x"]) for row in tail_rows], dtype=float)
    tail_x_finite = tail_x[np.isfinite(tail_x)]
    if tail_x_finite.size < 2:
        raise RuntimeError(f"Need at least two finite tail samples for {name}.")
    mean_x = float(np.mean(tail_x_finite))
    std_x = float(np.std(tail_x_finite, ddof=1))
    x_most = float(mode_x_from_samples(tail_x_finite, grid))
    window = EnsembleWindow(
        name=name,
        center_x=float(center_x),
        k=float(k),
        root=root,
        eq_file=eq_file,
        tail_file=tail_file,
        eq_rows=eq_rows,
        tail_rows=tail_rows,
        mean_x=mean_x,
        std_x=std_x,
        x_most=x_most,
        generation=int(generation),
        side=side,
    )
    write_json(root / "window_summary.json", build_window_summary(window, out_root))
    return window


def source_left_anchor(source: EQCluster | EnsembleWindow) -> float:
    if isinstance(source, EQCluster):
        return float(source.left_x)
    return float(source.mean_x)


def source_right_anchor(source: EQCluster | EnsembleWindow) -> float:
    if isinstance(source, EQCluster):
        return float(source.right_x)
    return float(source.mean_x)


def source_name(source: EQCluster | EnsembleWindow) -> str:
    return source.name


def source_boundary_window(
    source: EQCluster | EnsembleWindow,
    side: str,
) -> EnsembleWindow:
    if isinstance(source, EnsembleWindow):
        return source
    return source.windows[0] if side == "left" else source.windows[-1]


def pair_protocol_k(
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    k_min: float,
    k_max: float,
) -> float:
    return float(min(max(max(left_window.k, right_window.k), k_min), k_max))


def linear_path_centers(start_x: float, end_x: float, n_time: int) -> list[float]:
    if n_time <= 1:
        return [float(start_x), float(end_x)]
    return [float(value) for value in np.linspace(start_x, end_x, num=n_time)]


def window_tail_mean_sigma(window: "EnsembleWindow") -> tuple[float, float]:
    x = eq_tail_samples(window)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise RuntimeError(f"Need at least two finite tail samples for {window.name}.")
    mean = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(f"Invalid tail sigma for {window.name}: {sigma}")
    return mean, sigma


def eq_gt_tuple(window: "EnsembleWindow") -> tuple[float, float, float, float]:
    return (float(window.center_x), float(window.k), float(window.mean_x), float(window.std_x))


def get_k0_x0_harmonic_fromEQ(
    EQ_L: tuple[float, float, float, float],
    EQ_R: tuple[float, float, float, float],
) -> tuple[float, float, float, float, dict[str, Any]]:
    x_L, k_L, m_L, sigma_L = EQ_L
    x_R, k_R, m_R, sigma_R = EQ_R
    eps = 1.0e-12
    K_L = 1.0 / (float(sigma_L) ** 2)
    k0_L = K_L - float(k_L)
    left_x0_fallback = abs(k0_L) < eps
    x0_L = float(m_L) if left_x0_fallback else (K_L * float(m_L) - float(k_L) * float(x_L)) / k0_L
    K_R = 1.0 / (float(sigma_R) ** 2)
    k0_R = K_R - float(k_R)
    right_x0_fallback = abs(k0_R) < eps
    x0_R = float(m_R) if right_x0_fallback else (K_R * float(m_R) - float(k_R) * float(x_R)) / k0_R
    metadata: dict[str, Any] = {
        "x_L": float(x_L), "k_L": float(k_L), "m_L": float(m_L), "sigma_L": float(sigma_L),
        "K_L": float(K_L), "k0_L": float(k0_L), "x0_L": float(x0_L),
        "x_R": float(x_R), "k_R": float(k_R), "m_R": float(m_R), "sigma_R": float(sigma_R),
        "K_R": float(K_R), "k0_R": float(k0_R), "x0_R": float(x0_R),
        "left_x0_fallback": bool(left_x0_fallback),
        "right_x0_fallback": bool(right_x0_fallback),
    }
    return float(x0_L), float(k0_L), float(x0_R), float(k0_R), metadata


def estimate_mean_only_k0_x0_from_eq_pair(
    left_window: "EnsembleWindow",
    right_window: "EnsembleWindow",
    *,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    x_L = float(left_window.center_x)
    k_L = float(left_window.k)
    m_L = float(left_window.mean_x)
    x_R = float(right_window.center_x)
    k_R = float(right_window.k)
    m_R = float(right_window.mean_x)
    denom = m_L - m_R
    valid = True
    fallback_used = False
    fallback_reason = ""
    k0_segment = float("nan")
    x0_segment = float("nan")
    if abs(denom) < eps:
        valid = False
        fallback_used = True
        fallback_reason = "degenerate_means"
        k0_segment = float("nan")
        x0_segment = float("nan")
    else:
        numerator = k_R * (m_R - x_R) - k_L * (m_L - x_L)
        k0_segment = numerator / denom
        if abs(k0_segment) < eps:
            fallback_used = True
            fallback_reason = "near_zero_k0"
            x0_segment = 0.5 * (m_L + m_R)
        else:
            x0_segment = m_L + k_L * (m_L - x_L) / k0_segment
    lo = min(m_L, m_R)
    hi = max(m_L, m_R)
    is_transition = (
        valid
        and not fallback_used
        and math.isfinite(k0_segment)
        and math.isfinite(x0_segment)
        and k0_segment < 0.0
        and lo <= x0_segment <= hi
    )
    return {
        "k0_segment": float(k0_segment),
        "x0_segment": float(x0_segment),
        "valid": bool(valid),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "segment_type": "transition" if is_transition else "regular",
        "left_center_x": float(x_L),
        "right_center_x": float(x_R),
        "left_k": float(k_L),
        "right_k": float(k_R),
        "left_mean_x": float(m_L),
        "right_mean_x": float(m_R),
    }


def fit_quadratic_background_from_segment_patch(
    *,
    segment: "NEQSegment",
    patch: "PMFPatch",
    grid: np.ndarray,
    variance_floor: float,
    left_boundary: "EnsembleWindow",
    right_boundary: "EnsembleWindow",
    min_fit_bins: int = 5,
    k0_min_abs: float = 1.0e-8,
    x0_margin_factor: float = 0.25,
    bin_width: float = 0.1,
) -> dict[str, Any]:
    """Fit a weighted quadratic F(x)=a*x²+b*x+c to a segment-local NEQ PMF patch."""
    _nan = float("nan")
    _base = {
        "fit_accepted": False,
        "fit_source": "NEQ_quadratic_fit",
        "segment": segment.name,
        "patch_name": patch.name,
        "patch_kind": patch.kind,
        "n_fit_bins": 0,
        "x_fit_min": _nan, "x_fit_max": _nan,
        "k0": _nan, "x0": _nan, "F0": _nan,
        "a": _nan, "b": _nan, "c": _nan,
        "weighted_rmse": _nan, "reduced_chi2": _nan,
        "variance_floor": float(variance_floor),
        "fallback_reason": "",
    }
    coverage = np.asarray(patch.coverage_mask, dtype=bool)
    pmf_arr = np.asarray(patch.pmf, dtype=float)
    var_arr = np.asarray(patch.variance, dtype=float)
    grid_arr = np.asarray(grid, dtype=float)

    valid = coverage & np.isfinite(pmf_arr) & np.isfinite(var_arr)
    n_fit_bins = int(np.sum(valid))
    if n_fit_bins < min_fit_bins:
        return {**_base, "n_fit_bins": n_fit_bins,
                "fallback_reason": f"too_few_valid_bins:{n_fit_bins}<{min_fit_bins}"}

    x_fit = grid_arr[valid]
    f_fit = pmf_arr[valid]
    w_fit = 1.0 / (var_arr[valid] + float(variance_floor))

    sqrt_w = np.sqrt(w_fit)
    A = np.column_stack([x_fit ** 2, x_fit, np.ones(n_fit_bins)])
    Aw = A * sqrt_w[:, None]
    fw = f_fit * sqrt_w
    try:
        theta, _, _, _ = np.linalg.lstsq(Aw, fw, rcond=None)
    except Exception as exc:
        return {**_base, "n_fit_bins": n_fit_bins, "fallback_reason": f"lstsq_failed:{exc}"}

    a_val = float(theta[0])
    b_val = float(theta[1])
    c_val = float(theta[2])
    k0_fit = 2.0 * a_val

    if abs(k0_fit) < float(k0_min_abs):
        return {**_base, "n_fit_bins": n_fit_bins,
                "a": a_val, "b": b_val, "c": c_val, "k0": k0_fit,
                "fallback_reason": f"k0_below_min_abs:{abs(k0_fit):.3e}<{k0_min_abs}"}

    x0_fit = -b_val / (2.0 * a_val)
    F0_fit = c_val - 0.5 * k0_fit * x0_fit ** 2

    if not (np.isfinite(k0_fit) and np.isfinite(x0_fit) and np.isfinite(F0_fit)):
        return {**_base, "n_fit_bins": n_fit_bins,
                "a": a_val, "b": b_val, "c": c_val, "k0": k0_fit, "x0": x0_fit,
                "fallback_reason": "nonfinite_k0_x0_or_F0"}

    m_L = float(left_boundary.mean_x)
    m_R = float(right_boundary.mean_x)
    segment_width = abs(m_R - m_L)
    margin = max(2.0 * float(bin_width), float(x0_margin_factor) * segment_width)
    lo = min(m_L, m_R) - margin
    hi = max(m_L, m_R) + margin
    if not (lo <= x0_fit <= hi):
        return {**_base, "n_fit_bins": n_fit_bins,
                "a": a_val, "b": b_val, "c": c_val, "k0": k0_fit, "x0": x0_fit, "F0": F0_fit,
                "fallback_reason": f"x0_outside_margin:{x0_fit:.4f} not in [{lo:.4f},{hi:.4f}]"}

    residuals = A @ theta - f_fit
    weighted_ss = float(np.sum(w_fit * residuals ** 2))
    weighted_rmse = float(np.sqrt(weighted_ss / n_fit_bins)) if n_fit_bins > 0 else _nan
    reduced_chi2 = float(weighted_ss / max(n_fit_bins - 3, 1))

    if not np.isfinite(weighted_rmse):
        return {**_base, "n_fit_bins": n_fit_bins,
                "a": a_val, "b": b_val, "c": c_val, "k0": k0_fit, "x0": x0_fit, "F0": F0_fit,
                "weighted_rmse": weighted_rmse, "reduced_chi2": reduced_chi2,
                "fallback_reason": "nonfinite_weighted_rmse"}

    return {
        "fit_accepted": True,
        "fit_source": "NEQ_quadratic_fit",
        "segment": segment.name,
        "patch_name": patch.name,
        "patch_kind": patch.kind,
        "n_fit_bins": n_fit_bins,
        "x_fit_min": float(np.min(x_fit)),
        "x_fit_max": float(np.max(x_fit)),
        "k0": k0_fit,
        "x0": x0_fit,
        "F0": F0_fit,
        "a": a_val, "b": b_val, "c": c_val,
        "weighted_rmse": weighted_rmse,
        "reduced_chi2": reduced_chi2,
        "variance_floor": float(variance_floor),
        "fallback_reason": "",
    }


def get_neq_fit_for_rescue(
    *,
    target_bin_x: float,
    all_segments: "list[NEQSegment]",
    neq_patch_store: "dict[str, PMFPatch]",
    grid: np.ndarray,
    variance_floor: float,
    left_boundary: "EnsembleWindow",
    right_boundary: "EnsembleWindow",
    min_fit_bins: int = 5,
    k0_min_abs: float = 1.0e-8,
    x0_margin_factor: float = 0.25,
    bin_width: float = 0.1,
    ctx: "dict[str, Any] | None" = None,
    out_root: "Path | None" = None,
    n_boot: int = 16,
    rng_seed: int = 0,
) -> tuple["dict[str, Any]", "list[dict[str, Any]]"]:
    """Try segment-local NEQ patches to fit a quadratic background for rescue GT.

    Returns (best_fit_result, all_fit_rows). best_fit_result has fit_accepted=False if
    no patch produced an accepted fit.
    """
    covering = segment_covering_target(float(target_bin_x), all_segments)
    fit_rows: list[dict[str, Any]] = []
    best_fit: dict[str, Any] = {
        "fit_accepted": False,
        "fit_source": "",
        "segment": covering.name if covering else "",
        "patch_name": "",
        "patch_kind": "",
        "n_fit_bins": 0,
        "x_fit_min": float("nan"), "x_fit_max": float("nan"),
        "k0": float("nan"), "x0": float("nan"), "F0": float("nan"),
        "a": float("nan"), "b": float("nan"), "c": float("nan"),
        "weighted_rmse": float("nan"), "reduced_chi2": float("nan"),
        "variance_floor": float(variance_floor),
        "fallback_reason": "no_covering_segment" if covering is None else "no_valid_patch",
    }

    if covering is None:
        return best_fit, fit_rows

    _fit_kwargs = dict(
        segment=covering, grid=grid, variance_floor=variance_floor,
        left_boundary=left_boundary, right_boundary=right_boundary,
        min_fit_bins=min_fit_bins, k0_min_abs=k0_min_abs,
        x0_margin_factor=x0_margin_factor, bin_width=bin_width,
    )

    # 1. Try MTS patch from neq_patch_store (preferred: bidirectional, has bootstrap variance)
    mts_patch = neq_patch_store.get(covering.name)
    if mts_patch is not None and bool(covering.mts_patch_built):
        mts_fit = fit_quadratic_background_from_segment_patch(patch=mts_patch, **_fit_kwargs)
        mts_fit["round_patch_kind"] = "MTS"
        fit_rows.append(mts_fit)
        if mts_fit["fit_accepted"]:
            best_fit = mts_fit
            return best_fit, fit_rows

    # 2. Try HS fallback patches (forward then reverse) built on-the-fly via bootstrap
    if ctx is not None:
        _hs_root = (out_root / "rescue_neq_hs_fit" / covering.name) if out_root else Path("/tmp/hs_rescue_fit")
        for direction_idx, direction in enumerate(("forward", "reverse")):
            hs_patch, _ = bootstrap_hs_patch(
                segment=covering,
                direction=direction,
                grid=grid,
                ctx=ctx,
                n_boot=max(int(n_boot), 2),
                rng_seed=int(rng_seed) + direction_idx,
                out_root=out_root or _hs_root,
            )
            if hs_patch is not None:
                hs_fit = fit_quadratic_background_from_segment_patch(patch=hs_patch, **_fit_kwargs)
                hs_fit["round_patch_kind"] = f"HS_{direction.upper()}"
                fit_rows.append(hs_fit)
                if hs_fit["fit_accepted"]:
                    if not best_fit["fit_accepted"] or float(hs_fit["weighted_rmse"]) < float(best_fit["weighted_rmse"]):
                        best_fit = hs_fit

    return best_fit, fit_rows


def fit_quadratic_background_from_global_pmf(
    *,
    target_bin_x: float,
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: "np.ndarray | None",
    variance_floor: float,
    left_boundary: "EnsembleWindow",
    right_boundary: "EnsembleWindow",
    min_fit_bins: int = 5,
    k0_min_abs: float = 1.0e-8,
    x0_margin_factor: float = 0.25,
    bin_width: float = 0.1,
    fit_radius: "float | None" = None,
) -> dict[str, Any]:
    """Fit a weighted quadratic F(x)=a*x²+b*x+c to the global fused PMF near target_bin_x."""
    _nan = float("nan")
    _base = {
        "fit_accepted": False,
        "fit_source": "global_pmf_quadratic_fit",
        "segment": "",
        "patch_name": "global_pmf",
        "patch_kind": "GLOBAL_FUSED_PMF",
        "n_fit_bins": 0,
        "x_fit_min": _nan, "x_fit_max": _nan,
        "k0": _nan, "x0": _nan, "F0": _nan,
        "a": _nan, "b": _nan, "c": _nan,
        "weighted_rmse": _nan, "reduced_chi2": _nan,
        "variance_floor": float(variance_floor),
        "fallback_reason": "",
    }

    m_L = float(left_boundary.mean_x)
    m_R = float(right_boundary.mean_x)
    t_x = float(target_bin_x)
    segment_width = max(abs(m_R - m_L), float(bin_width))
    margin = max(2.0 * float(bin_width), float(x0_margin_factor) * segment_width)
    lo = min(m_L, m_R, t_x) - margin
    hi = max(m_L, m_R, t_x) + margin
    if fit_radius is not None:
        lo = max(lo, t_x - float(fit_radius))
        hi = min(hi, t_x + float(fit_radius))

    grid_arr = np.asarray(grid, dtype=float)
    pmf_arr = np.asarray(global_pmf, dtype=float)
    domain_mask = (grid_arr >= lo) & (grid_arr <= hi) & np.isfinite(pmf_arr)

    n_fit_bins = int(np.sum(domain_mask))
    if n_fit_bins < min_fit_bins:
        return {**_base, "n_fit_bins": n_fit_bins,
                "fallback_reason": f"too_few_valid_bins:{n_fit_bins}<{min_fit_bins}"}

    x_fit = grid_arr[domain_mask]
    f_fit = pmf_arr[domain_mask]

    if global_variance is not None:
        var_arr = np.asarray(global_variance, dtype=float)
        var_local = var_arr[domain_mask]
        has_var = np.isfinite(var_local)
        if np.any(has_var):
            w_fit = np.where(has_var, 1.0 / (var_local + float(variance_floor)), 1.0 / float(variance_floor))
        else:
            w_fit = np.full(n_fit_bins, 1.0 / float(variance_floor))
    else:
        w_fit = np.full(n_fit_bins, 1.0 / float(variance_floor))

    sqrt_w = np.sqrt(w_fit)
    A = np.column_stack([x_fit ** 2, x_fit, np.ones(n_fit_bins)])
    Aw = A * sqrt_w[:, None]
    fw = f_fit * sqrt_w
    try:
        theta, _, _, _ = np.linalg.lstsq(Aw, fw, rcond=None)
    except Exception as exc:
        return {**_base, "n_fit_bins": n_fit_bins, "fallback_reason": f"lstsq_failed:{exc}"}

    a_val, b_val, c_val = float(theta[0]), float(theta[1]), float(theta[2])
    k0_fit = 2.0 * a_val

    if abs(k0_fit) < float(k0_min_abs):
        return {**_base, "n_fit_bins": n_fit_bins, "a": a_val, "b": b_val, "c": c_val, "k0": k0_fit,
                "fallback_reason": f"k0_below_min_abs:{abs(k0_fit):.3e}<{k0_min_abs}"}

    x0_fit = -b_val / (2.0 * a_val)
    F0_fit = c_val - 0.5 * k0_fit * x0_fit ** 2

    if not (np.isfinite(k0_fit) and np.isfinite(x0_fit) and np.isfinite(F0_fit)):
        return {**_base, "n_fit_bins": n_fit_bins, "a": a_val, "b": b_val, "c": c_val,
                "k0": k0_fit, "x0": x0_fit, "fallback_reason": "nonfinite_k0_x0_or_F0"}

    # x0 sanity: must be within margin of the local segment
    seg_width2 = abs(m_R - m_L)
    margin2 = max(2.0 * float(bin_width), float(x0_margin_factor) * max(seg_width2, float(bin_width)))
    lo_x0 = min(m_L, m_R, t_x) - margin2
    hi_x0 = max(m_L, m_R, t_x) + margin2
    if not (lo_x0 <= x0_fit <= hi_x0):
        return {**_base, "n_fit_bins": n_fit_bins, "a": a_val, "b": b_val, "c": c_val,
                "k0": k0_fit, "x0": x0_fit, "F0": F0_fit,
                "fallback_reason": f"x0_outside_margin:{x0_fit:.4f} not in [{lo_x0:.4f},{hi_x0:.4f}]"}

    residuals = A @ theta - f_fit
    weighted_ss = float(np.sum(w_fit * residuals ** 2))
    weighted_rmse = float(np.sqrt(weighted_ss / n_fit_bins)) if n_fit_bins > 0 else _nan
    reduced_chi2 = float(weighted_ss / max(n_fit_bins - 3, 1))

    if not np.isfinite(weighted_rmse):
        return {**_base, "n_fit_bins": n_fit_bins, "a": a_val, "b": b_val, "c": c_val,
                "k0": k0_fit, "x0": x0_fit, "F0": F0_fit,
                "weighted_rmse": weighted_rmse, "reduced_chi2": reduced_chi2,
                "fallback_reason": "nonfinite_weighted_rmse"}

    return {
        "fit_accepted": True,
        "fit_source": "global_pmf_quadratic_fit",
        "segment": "",
        "patch_name": "global_pmf",
        "patch_kind": "GLOBAL_FUSED_PMF",
        "n_fit_bins": n_fit_bins,
        "x_fit_min": float(np.min(x_fit)),
        "x_fit_max": float(np.max(x_fit)),
        "k0": k0_fit,
        "x0": x0_fit,
        "F0": F0_fit,
        "a": a_val, "b": b_val, "c": c_val,
        "weighted_rmse": weighted_rmse,
        "reduced_chi2": reduced_chi2,
        "variance_floor": float(variance_floor),
        "fallback_reason": "",
    }


def midpoint_xs_ks_from_EQ(
    EQ_L: tuple[float, float, float, float],
    EQ_R: tuple[float, float, float, float],
    k_bound: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    x_L, k_L, _m_L, _sigma_L = EQ_L
    x_R, k_R, _m_R, _sigma_R = EQ_R
    k_min, k_max = k_bound
    x_mid = 0.5 * (float(x_L) + float(x_R))
    k_mid_raw = (0.5 * (math.sqrt(float(k_L)) + math.sqrt(float(k_R)))) ** 2
    k_mid = float(min(max(k_mid_raw, float(k_min)), float(k_max)))
    metadata: dict[str, Any] = {
        "method": "midpoint_fallback",
        "x_mid": float(x_mid),
        "k_mid_raw": float(k_mid_raw),
        "k_mid": float(k_mid),
    }
    return float(x_mid), float(k_mid), metadata


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
    x_L, k_L, m_L, sigma_L = EQ_L
    x_R, k_R, m_R, sigma_R = EQ_R
    k_min, k_max = k_bound
    x_low = min(float(x_L), float(x_R))
    x_high = max(float(x_L), float(x_R))
    s = float(s)
    if s <= 0.0:
        x_s, k_s = float(x_L), float(k_L)
        raw_x, raw_k = float(x_L), float(k_L)
        x_clipped = k_clipped = False
        m_target = float(m_L)
        sigma_target = float(sigma_L)
        K_target = 1.0 / (float(sigma_L) ** 2)
        x0_s = float(x0_L)
        k0_s = float(k0_L)
    elif s >= 1.0:
        x_s, k_s = float(x_R), float(k_R)
        raw_x, raw_k = float(x_R), float(k_R)
        x_clipped = k_clipped = False
        m_target = float(m_R)
        sigma_target = float(sigma_R)
        K_target = 1.0 / (float(sigma_R) ** 2)
        x0_s = float(x0_R)
        k0_s = float(k0_R)
    else:
        x0_s = (1.0 - s) * float(x0_L) + s * float(x0_R)
        k0_s = (1.0 - s) * float(k0_L) + s * float(k0_R)
        m_target = (1.0 - s) * float(m_L) + s * float(m_R)
        sigma_target = (1.0 - s) * float(sigma_L) + s * float(sigma_R)
        K_target = 1.0 / (sigma_target ** 2)
        raw_k = K_target - k0_s
        k_s, k_clipped = clip_with_flag(raw_k, float(k_min), float(k_max))
        raw_x = ((k0_s + k_s) * m_target - k0_s * x0_s) / k_s
        x_s, x_clipped = clip_with_flag(raw_x, x_low, x_high)
    metadata: dict[str, Any] = {
        "method": "GT",
        "s": s,
        "x0_s": float(x0_s),
        "k0_s": float(k0_s),
        "m_target": float(m_target),
        "sigma_target": float(sigma_target),
        "K_target": float(K_target),
        "k_raw": float(raw_k),
        "k": float(k_s),
        "k_clipped": bool(k_clipped),
        "x_raw": float(raw_x),
        "x": float(x_s),
        "x_clipped": bool(x_clipped),
    }
    return float(x_s), float(k_s), metadata


def get_xs_ks_from_s_mean_only(
    k0_segment: float,
    x0_segment: float,
    s: float,
    EQ_L: tuple[float, float, float, float],
    EQ_R: tuple[float, float, float, float],
    k_bound: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    x_L, k_L, m_L, sigma_L = EQ_L
    x_R, k_R, m_R, sigma_R = EQ_R
    k_min, k_max = k_bound
    x_low = min(float(x_L), float(x_R))
    x_high = max(float(x_L), float(x_R))
    s = float(s)
    m_target = (1.0 - s) * float(m_L) + s * float(m_R)
    sigma_target = (1.0 - s) * float(sigma_L) + s * float(sigma_R)
    K_target = 1.0 / max(sigma_target ** 2, 1.0e-18)
    k_raw = K_target - float(k0_segment)
    k_s, k_clipped = clip_with_flag(k_raw, float(k_min), float(k_max))
    if abs(k_s) < 1.0e-18:
        raw_x = float(x_L) if s <= 0.5 else float(x_R)
        x_s, x_clipped = raw_x, False
    else:
        raw_x = ((float(k0_segment) + k_s) * m_target - float(k0_segment) * float(x0_segment)) / k_s
        x_s, x_clipped = clip_with_flag(raw_x, x_low, x_high)
    return float(x_s), float(k_s), {
        "mode": "GT_mean_only",
        "s": s,
        "k0_segment": float(k0_segment),
        "x0_segment": float(x0_segment),
        "m_target": float(m_target),
        "sigma_target": float(sigma_target),
        "K_target": float(K_target),
        "k_raw": float(k_raw),
        "k": float(k_s),
        "k_clipped": bool(k_clipped),
        "x_raw": float(raw_x),
        "x": float(x_s),
        "x_clipped": bool(x_clipped),
    }


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
    x_L, k_L, m_L, sigma_L = EQ_L
    x_R, k_R, m_R, sigma_R = EQ_R
    k_min, k_max = k_bound
    x_low = min(float(x_L), float(x_R))
    x_high = max(float(x_L), float(x_R))
    if abs(float(m_R) - float(m_L)) < 1.0e-12:
        raise RuntimeError(
            f"get_xs_ks_from_ms: m_L ({m_L}) == m_R ({m_R}); cannot compute s_eff."
        )
    s_eff = (float(ms) - float(m_L)) / (float(m_R) - float(m_L))
    x0_s = k0_s = sigma_target = K_target = k_raw = k_s = x_raw = x_s = float("nan")
    k_clipped = False
    used_midpoint_fallback = False
    fallback_reason = ""
    method = "GT_ms"
    if s_eff <= 0.0 or s_eff >= 1.0:
        used_midpoint_fallback = True
        fallback_reason = "s_eff_endpoint"
        x_s, k_s, mid_meta = midpoint_xs_ks_from_EQ(EQ_L, EQ_R, k_bound)
        x_raw = x_s
        k_raw = float(mid_meta["k_mid_raw"])
        method = "midpoint_fallback"
    else:
        x0_s = (1.0 - s_eff) * float(x0_L) + s_eff * float(x0_R)
        k0_s = (1.0 - s_eff) * float(k0_L) + s_eff * float(k0_R)
        sigma_target = (1.0 - s_eff) * float(sigma_L) + s_eff * float(sigma_R)
        K_target = 1.0 / (sigma_target ** 2)
        k_raw = K_target - k0_s
        k_s, k_clipped = clip_with_flag(k_raw, float(k_min), float(k_max))
        x_raw = ((k0_s + k_s) * float(ms) - k0_s * x0_s) / k_s
        if x_raw < x_low or x_raw > x_high:
            used_midpoint_fallback = True
            fallback_reason = "x_raw_out_of_bounds"
            x_s, k_s, mid_meta = midpoint_xs_ks_from_EQ(EQ_L, EQ_R, k_bound)
            k_raw = float(mid_meta["k_mid_raw"])
            method = "midpoint_fallback"
        else:
            x_s = x_raw
    metadata: dict[str, Any] = {
        "method": method,
        "ms": float(ms),
        "s_eff": float(s_eff),
        "x0_s": float(x0_s),
        "k0_s": float(k0_s),
        "sigma_target": float(sigma_target),
        "K_target": float(K_target),
        "k_raw": float(k_raw),
        "k": float(k_s),
        "k_clipped": bool(k_clipped),
        "x_raw": float(x_raw),
        "x": float(x_s),
        "used_midpoint_fallback": bool(used_midpoint_fallback),
        "fallback_reason": fallback_reason,
    }
    return float(x_s), float(k_s), metadata


def clip_with_flag(value: float, lower: float, upper: float) -> tuple[float, bool]:
    clipped = float(min(max(float(value), float(lower)), float(upper)))
    was_clipped = bool(abs(clipped - float(value)) > 1.0e-12)
    return clipped, was_clipped


def build_linear_bridge_protocol(
    *,
    left_window: "EnsembleWindow",
    right_window: "EnsembleWindow",
    n_time: int,
    k_min: float,
    k_max: float,
) -> dict[str, Any]:
    if n_time <= 1:
        n_time = 2
    xL = float(left_window.center_x)
    xR = float(right_window.center_x)
    kL = float(left_window.k)
    kR = float(right_window.k)
    x_low = min(xL, xR)
    x_high = max(xL, xR)
    s_values = np.linspace(0.0, 1.0, num=int(n_time))
    centers: list[float] = []
    ks: list[float] = []
    rows: list[dict[str, Any]] = []
    n_clip_x = 0
    n_clip_k = 0
    for idx, s in enumerate(s_values):
        if idx == 0:
            x_s, k_s, raw_x, raw_k = xL, kL, xL, kL
            clipped_x = clipped_k = False
        elif idx == len(s_values) - 1:
            x_s, k_s, raw_x, raw_k = xR, kR, xR, kR
            clipped_x = clipped_k = False
        else:
            raw_x = (1.0 - float(s)) * xL + float(s) * xR
            sqrt_k = (1.0 - float(s)) * math.sqrt(kL) + float(s) * math.sqrt(kR)
            raw_k = sqrt_k * sqrt_k
            x_s, clipped_x = clip_with_flag(raw_x, x_low, x_high)
            k_s, clipped_k = clip_with_flag(raw_k, k_min, k_max)
        n_clip_x += int(clipped_x)
        n_clip_k += int(clipped_k)
        centers.append(float(x_s))
        ks.append(float(k_s))
        rows.append({
            "step_index": int(idx),
            "s": float(s),
            "mode": "linear",
            "x_raw": float(raw_x),
            "k_raw": float(raw_k),
            "x": float(x_s),
            "k": float(k_s),
            "x_clipped": int(clipped_x),
            "k_clipped": int(clipped_k),
            "m_target": "",
            "sigma_target": "",
            "m_actual": "",
            "k0_local": "",
            "q_local": "",
        })
    n = len(s_values)
    return {
        "centers": centers,
        "ks": ks,
        "rows": rows,
        "metadata": {
            "protocol_mode": "linear",
            "x_min": float(x_low),
            "x_max": float(x_high),
            "k_min": float(k_min),
            "k_max": float(k_max),
            "clip_fraction_x": float(n_clip_x) / float(n),
            "clip_fraction_k": float(n_clip_k) / float(n),
        },
    }


def build_gt_bridge_protocol(
    *,
    left_window: "EnsembleWindow",
    right_window: "EnsembleWindow",
    n_time: int,
    k_min: float,
    k_max: float,
) -> dict[str, Any]:
    if n_time <= 1:
        n_time = 2
    EQ_L = eq_gt_tuple(left_window)
    EQ_R = eq_gt_tuple(right_window)
    xL, kL, mL, sigmaL = EQ_L
    xR, kR, mR, sigmaR = EQ_R
    x_low = min(float(xL), float(xR))
    x_high = max(float(xL), float(xR))
    # Single mean-only harmonic background for the whole segment
    mo = estimate_mean_only_k0_x0_from_eq_pair(left_window, right_window)
    k0_segment = float(mo["k0_segment"])
    x0_segment = float(mo["x0_segment"])
    segment_type = str(mo["segment_type"])
    s_values = np.linspace(0.0, 1.0, num=int(n_time))
    centers: list[float] = []
    ks: list[float] = []
    rows: list[dict[str, Any]] = []
    n_clip_x = 0
    n_clip_k = 0
    for idx, s in enumerate(s_values):
        x_s, k_s, gt_meta = get_xs_ks_from_s_mean_only(
            k0_segment, x0_segment, float(s), EQ_L, EQ_R, (float(k_min), float(k_max))
        )
        clipped_x = bool(gt_meta["x_clipped"])
        clipped_k = bool(gt_meta["k_clipped"])
        n_clip_x += int(clipped_x)
        n_clip_k += int(clipped_k)
        centers.append(float(x_s))
        ks.append(float(k_s))
        rows.append({
            "step_index": int(idx),
            "s": float(s),
            "mode": "GT_mean_only",
            "x_raw": float(gt_meta["x_raw"]),
            "k_raw": float(gt_meta["k_raw"]),
            "x": float(x_s),
            "k": float(k_s),
            "x_clipped": int(clipped_x),
            "k_clipped": int(clipped_k),
            "m_target": float(gt_meta["m_target"]),
            "sigma_target": float(gt_meta["sigma_target"]),
            "K_target": float(gt_meta["K_target"]),
            "k0_segment": float(k0_segment),
            "x0_segment": float(x0_segment),
            "segment_type": segment_type,
            "mL": float(mL),
            "sigmaL": float(sigmaL),
            "mR": float(mR),
            "sigmaR": float(sigmaR),
        })
    n = len(s_values)
    return {
        "centers": centers,
        "ks": ks,
        "rows": rows,
        "metadata": {
            "protocol_mode": "GT_mean_only",
            "x_min": float(x_low),
            "x_max": float(x_high),
            "k_min": float(k_min),
            "k_max": float(k_max),
            "clip_fraction_x": float(n_clip_x) / float(n),
            "clip_fraction_k": float(n_clip_k) / float(n),
            "mL": float(mL),
            "sigmaL": float(sigmaL),
            "mR": float(mR),
            "sigmaR": float(sigmaR),
            "k0_segment": float(k0_segment),
            "x0_segment": float(x0_segment),
            "segment_type": segment_type,
            **{f"mo_{k}": v for k, v in mo.items()},
        },
    }


def build_bridge_protocol(
    *,
    mode: str,
    left_window: "EnsembleWindow",
    right_window: "EnsembleWindow",
    n_time: int,
    k_min: float,
    k_max: float,
) -> dict[str, Any]:
    normalized = str(mode).upper()
    if normalized == "GT":
        return build_gt_bridge_protocol(
            left_window=left_window,
            right_window=right_window,
            n_time=n_time,
            k_min=k_min,
            k_max=k_max,
        )
    if normalized == "LINEAR":
        return build_linear_bridge_protocol(
            left_window=left_window,
            right_window=right_window,
            n_time=n_time,
            k_min=k_min,
            k_max=k_max,
        )
    raise ValueError(f"Unknown NEQ protocol mode: {mode!r}")


def run_neq_edge_with_protocol_path(
    *,
    bin_path: str,
    ctx: dict[str, Any],
    left_center: float,
    right_center: float,
    eq_left: Path,
    eq_right: Path,
    protocol_path: Path,
    k_fallback: float,
    n_traj_per_direction: int,
    t_neq: int,
    nout: int,
    seed: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        bin_path,
        *build_common_args(ctx),
        "-k", str(k_fallback),
        "-k_midscale", str(ctx["nes_screen"]["fixed"]["k_midscale"]),
        "-A_center", f"{left_center},0.0",
        "-B_center", f"{right_center},0.0",
        "-eq0", str(eq_left),
        "-eq1", str(eq_right),
        "-fpath", str(protocol_path),
        "-N_neq", str(n_traj_per_direction),
        "-T_neq", str(t_neq),
        "-neq_nout", str(nout),
        "-neq_seed", str(seed),
        "-out_dir", str(out_dir),
        "-log", str(out_dir / "neq.log"),
    ]
    run_checked(cmd)


def run_neq_protocol(
    *,
    name: str,
    left: EQCluster | EnsembleWindow,
    right: EQCluster | EnsembleWindow,
    boundary_left: EnsembleWindow,
    boundary_right: EnsembleWindow,
    bin_path: str,
    ctx: dict[str, Any],
    t_neq: int,
    n_neq_traj: int,
    seed: int,
    root: Path,
    k_min: float,
    k_max: float,
    out_root: Path,
    neq_protocol_mode: str = "GT",
    n_neq_traj_requested: int | None = None,
    neq_budget_limited: bool = False,
    neq_cost_requested: int | None = None,
    neq_cost_actual: int | None = None,
    remaining_budget_before_segment: int | None = None,
) -> NEQSegment:
    root.mkdir(parents=True, exist_ok=True)
    protocol_root = root / "protocols"
    forward_root = root / "forward"
    reverse_root = root / "reverse"
    protocol_root.mkdir(parents=True, exist_ok=True)
    forward_root.mkdir(parents=True, exist_ok=True)
    reverse_root.mkdir(parents=True, exist_ok=True)

    n_time_requested = int(max(t_neq, 2))
    forward_protocol = build_bridge_protocol(
        mode=neq_protocol_mode,
        left_window=boundary_left,
        right_window=boundary_right,
        n_time=n_time_requested,
        k_min=float(k_min),
        k_max=float(k_max),
    )
    reverse_protocol = build_bridge_protocol(
        mode=neq_protocol_mode,
        left_window=boundary_right,
        right_window=boundary_left,
        n_time=n_time_requested,
        k_min=float(k_min),
        k_max=float(k_max),
    )

    forward_path_file = protocol_root / "forward_path.csv"
    reverse_path_file = protocol_root / "reverse_path.csv"
    write_protocol_path(forward_path_file, forward_protocol["centers"], forward_protocol["ks"])
    write_protocol_path(reverse_path_file, reverse_protocol["centers"], reverse_protocol["ks"])
    write_csv(
        protocol_root / "forward_protocol_diagnostics.csv",
        ordered_fieldnames(forward_protocol["rows"]),
        forward_protocol["rows"],
    )
    write_csv(
        protocol_root / "reverse_protocol_diagnostics.csv",
        ordered_fieldnames(reverse_protocol["rows"]),
        reverse_protocol["rows"],
    )
    write_json(
        protocol_root / "protocol_summary.json",
        {
            "mode": str(forward_protocol["metadata"]["protocol_mode"]),
            "forward": forward_protocol["metadata"],
            "reverse": reverse_protocol["metadata"],
        },
    )

    k_fallback = pair_protocol_k(boundary_left, boundary_right, k_min, k_max)
    run_neq_edge_with_protocol_path(
        bin_path=bin_path,
        ctx=ctx,
        left_center=float(boundary_left.center_x),
        right_center=float(boundary_right.center_x),
        eq_left=boundary_left.eq_file,
        eq_right=boundary_right.eq_file,
        protocol_path=forward_path_file,
        k_fallback=float(k_fallback),
        n_traj_per_direction=int(n_neq_traj),
        t_neq=int(t_neq),
        nout=int(max(t_neq, 1)),
        seed=int(seed),
        out_dir=root,
    )
    normalize_flat_neq_outputs_to_segment_layout(root)
    forward_files, reverse_files = discover_neq_trajectories(root)
    forward_trajectories = [read_csv_rows(path) for path in forward_files]
    reverse_trajectories = [read_csv_rows(path) for path in reverse_files]
    if not forward_trajectories or not reverse_trajectories:
        raise RuntimeError(f"NEQ segment {name} wrote no forward/reverse trajectories.")
    n_time = min(len(rows) for rows in forward_trajectories + reverse_trajectories)

    forward_centers = forward_protocol["centers"][:n_time]
    forward_ks = forward_protocol["ks"][:n_time]
    reverse_centers = reverse_protocol["centers"][:n_time]
    reverse_ks = reverse_protocol["ks"][:n_time]
    if n_time < n_time_requested:
        write_protocol_path(forward_path_file, forward_centers, forward_ks)
        write_protocol_path(reverse_path_file, reverse_centers, reverse_ks)

    forward_draws_file, forward_endpoints_file = summarize_trajectory_rows(
        forward_files,
        forward_trajectories,
        forward_root,
        "forward_endpoints.csv",
        out_root,
    )
    reverse_draws_file, reverse_endpoints_file = summarize_trajectory_rows(
        reverse_files,
        reverse_trajectories,
        reverse_root,
        "reverse_endpoints.csv",
        out_root,
    )
    fwd_arr = np.asarray(forward_ks, dtype=float)
    fwd_x_arr = np.asarray(forward_centers, dtype=float)
    protocol_k_mean = float(np.nanmean(fwd_arr))
    segment = NEQSegment(
        name=name,
        left=left,
        right=right,
        left_boundary=boundary_left,
        right_boundary=boundary_right,
        root=root,
        forward_trajectories=forward_trajectories,
        reverse_trajectories=reverse_trajectories,
        forward_trajectory_files=forward_files,
        reverse_trajectory_files=reverse_files,
        forward_path_file=forward_path_file,
        reverse_path_file=reverse_path_file,
        protocol_k=protocol_k_mean,
        n_neq_traj_requested=int(n_neq_traj if n_neq_traj_requested is None else n_neq_traj_requested),
        n_neq_traj_actual=int(n_neq_traj),
        neq_budget_limited=bool(neq_budget_limited),
        neq_cost_requested=int(stage_cost_neq(int(n_neq_traj if n_neq_traj_requested is None else n_neq_traj_requested), int(t_neq)) if neq_cost_requested is None else neq_cost_requested),
        neq_cost_actual=int(stage_cost_neq(int(n_neq_traj), int(t_neq)) if neq_cost_actual is None else neq_cost_actual),
        remaining_budget_before_segment=int(remaining_budget_before_segment or 0),
        protocol_mode=str(forward_protocol["metadata"]["protocol_mode"]),
        protocol_metadata={
            "forward": forward_protocol["metadata"],
            "reverse": reverse_protocol["metadata"],
        },
        protocol_k_min=float(np.nanmin(fwd_arr)),
        protocol_k_max=float(np.nanmax(fwd_arr)),
        protocol_x_min=float(np.nanmin(fwd_x_arr)),
        protocol_x_max=float(np.nanmax(fwd_x_arr)),
        protocol_clip_fraction_k=float(forward_protocol["metadata"]["clip_fraction_k"]),
        protocol_clip_fraction_x=float(forward_protocol["metadata"]["clip_fraction_x"]),
    )
    write_json(
        root / "segment_summary.json",
        {
            "name": name,
            "left": source_name(left),
            "right": source_name(right),
            "boundary_left": boundary_left.name,
            "boundary_right": boundary_right.name,
            "protocol_k": protocol_k_mean,
            "protocol_mode": segment.protocol_mode,
            "protocol_k_mean": protocol_k_mean,
            "protocol_k_min_observed": segment.protocol_k_min,
            "protocol_k_max_observed": segment.protocol_k_max,
            "protocol_x_min_observed": segment.protocol_x_min,
            "protocol_x_max_observed": segment.protocol_x_max,
            "protocol_clip_fraction_k": segment.protocol_clip_fraction_k,
            "protocol_clip_fraction_x": segment.protocol_clip_fraction_x,
            "protocol_summary_file": relative_to_root(protocol_root / "protocol_summary.json", out_root),
            "forward_protocol_diagnostics_file": relative_to_root(protocol_root / "forward_protocol_diagnostics.csv", out_root),
            "reverse_protocol_diagnostics_file": relative_to_root(protocol_root / "reverse_protocol_diagnostics.csv", out_root),
            "t_neq": int(t_neq),
            "n_neq_traj": int(n_neq_traj),
            "n_neq_traj_requested": int(segment.n_neq_traj_requested),
            "n_neq_traj_actual": int(segment.n_neq_traj_actual),
            "neq_budget_limited": bool(segment.neq_budget_limited),
            "neq_cost_requested": int(segment.neq_cost_requested),
            "neq_cost_actual": int(segment.neq_cost_actual),
            "remaining_budget_before_segment": int(segment.remaining_budget_before_segment),
            "n_forward": int(len(forward_trajectories)),
            "n_reverse": int(len(reverse_trajectories)),
            "forward_path_file": relative_to_root(forward_path_file, out_root),
            "reverse_path_file": relative_to_root(reverse_path_file, out_root),
            "forward_drawn_start_samples_file": relative_to_root(forward_draws_file, out_root),
            "forward_endpoints_file": relative_to_root(forward_endpoints_file, out_root),
            "reverse_drawn_start_samples_file": relative_to_root(reverse_draws_file, out_root),
            "reverse_endpoints_file": relative_to_root(reverse_endpoints_file, out_root),
            "forward_dir": relative_to_root(forward_root, out_root),
            "reverse_dir": relative_to_root(reverse_root, out_root),
            "trajectory_layout": "canonical_forward_reverse_traj_dirs",
        },
    )
    return segment


def cluster_name_from_windows(windows: list[EnsembleWindow]) -> str:
    if len(windows) == 1:
        return f"EQ_{windows[0].name}"
    return f"EQ_{windows[0].name}__{windows[-1].name}"


def build_eq_clusters(
    windows: list[EnsembleWindow],
    grid: np.ndarray,
    js_threshold: float,
    ctx: dict[str, Any] | None = None,
) -> tuple[list[EQCluster], list[dict[str, Any]]]:
    ordered = sorted(windows, key=lambda row: (float(row.mean_x), str(row.name)))
    if not ordered:
        return [], []
    _ctx: dict[str, Any] = ctx if ctx is not None else {}
    clusters: list[EQCluster] = []
    js_rows: list[dict[str, Any]] = []
    current = [ordered[0]]
    for window in ordered[1:]:
        left_window = current[-1]
        js_metrics = pair_jsd_metrics(
            eq_tail_samples(left_window),
            eq_tail_samples(window),
            grid,
        )
        js_norm = float(js_metrics["pair_jsd_norm"])
        bar_summary = compute_eq_pair_bar_summary(left_window, window, _ctx)
        js_rows.append(
            {
                "left_window": left_window.name,
                "right_window": window.name,
                "left_mean_x": float(left_window.mean_x),
                "right_mean_x": float(window.mean_x),
                "left_center_x": float(left_window.center_x),
                "right_center_x": float(window.center_x),
                **js_metrics,
                "js_threshold": float(js_threshold),
                "merged": bool(math.isfinite(js_norm) and js_norm <= float(js_threshold)),
                "cluster_order_coordinate": "mean_x",
                **bar_summary,
            }
        )
        if math.isfinite(js_norm) and js_norm <= float(js_threshold):
            current.append(window)
        else:
            cluster_windows = sorted(current, key=lambda w: (float(w.mean_x), str(w.name)))
            clusters.append(
                EQCluster(
                    name=cluster_name_from_windows(cluster_windows),
                    windows=cluster_windows,
                    left_x=float(min(w.mean_x for w in cluster_windows)),
                    right_x=float(max(w.mean_x for w in cluster_windows)),
                )
            )
            current = [window]
    cluster_windows = sorted(current, key=lambda w: (float(w.mean_x), str(w.name)))
    clusters.append(
        EQCluster(
            name=cluster_name_from_windows(cluster_windows),
            windows=cluster_windows,
            left_x=float(min(w.mean_x for w in cluster_windows)),
            right_x=float(max(w.mean_x for w in cluster_windows)),
        )
    )
    return clusters, js_rows


def build_eq_cluster_patch(
    cluster: EQCluster,
    grid: np.ndarray,
    ctx: dict[str, Any],
    n_boot: int,
    patch_root: Path,
    rng_seed: int,
) -> PMFPatch:
    patch_dir = patch_root / cluster.name
    window_rows = [
        {
            "tail_x": eq_tail_samples(window),
            "x_m": float(window.center_x),
            "k_m": float(window.k),
            "name": window.name,
        }
        for window in cluster.windows
    ]
    base_pmf, ess, probability = direct_eq_mbar_pmf(window_rows, grid, ctx)
    base_pmf = shift_finite_to_min_zero(base_pmf)
    analytic = analytic_pmf(grid, ctx)
    variance_stack = []
    anchor_columns: dict[str, np.ndarray] = {}
    boot_used_by_anchor: dict[str, int] = {}
    for anchor_idx, window in enumerate(cluster.windows):
        boot = bootstrap_direct_eq_mbar(
            window_rows,
            grid,
            float(window.mean_x),
            int(rng_seed + 31 * anchor_idx),
            ctx,
            int(n_boot),
        )
        boot_var = np.asarray(boot["boot_var"], dtype=float)
        variance_stack.append(boot_var)
        anchor_columns[window.name] = boot_var
        boot_used_by_anchor[window.name] = int(boot["n_boot_used"])
    variance = (
        np.nanmin(np.vstack(variance_stack), axis=0)
        if variance_stack
        else np.full(len(grid), np.nan, dtype=float)
    )
    coverage = np.isfinite(base_pmf) & np.isfinite(variance)
    coverage &= coverage_mask_from_samples(
        np.concatenate([eq_tail_samples(window) for window in cluster.windows]),
        grid,
    )
    anchor_variances = {
        f"var_anchor_{window_name}": variance_values
        for window_name, variance_values in anchor_columns.items()
    }
    anchor_variances["var_eq_min"] = variance
    patch = PMFPatch(
        name=cluster.name,
        kind="EQ_MBAR",
        root=patch_dir,
        grid=np.asarray(grid, dtype=float),
        pmf=base_pmf,
        variance=variance,
        coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[window.name for window in cluster.windows],
        metadata={
            "cluster_name": cluster.name,
            "window_names": [window.name for window in cluster.windows],
            "left_x": float(cluster.left_x),
            "right_x": float(cluster.right_x),
            "n_boot": int(n_boot),
            "mean_ess": float(np.nanmean(np.asarray(ess, dtype=float))) if len(ess) else None,
            "mean_probability": (
                float(np.nanmean(np.asarray(probability, dtype=float))) if len(probability) else None
            ),
            "anchor_boot_used": boot_used_by_anchor,
        },
        anchor_variances=anchor_variances,
    )
    write_patch_outputs(patch, analytic)
    return patch


def read_protocol_centers_and_k(path_file: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path_file)
    centers = np.asarray([float(row["x0"]) for row in rows], dtype=float)
    k_values = np.asarray([float(row["k"]) for row in rows], dtype=float)
    return centers, k_values


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


def build_neq_mts_patch(
    segment: NEQSegment,
    grid: np.ndarray,
    ctx: dict[str, Any],
    n_boot: int,
    patch_root: Path,
    rng_seed: int,
    *,
    disable_fixed_cft_bootstrap: bool = False,
) -> PMFPatch:
    patch_dir = patch_root / segment.name
    forward_frames = [pd.DataFrame(rows) for rows in segment.forward_trajectories]
    reverse_frames = [pd.DataFrame(rows) for rows in segment.reverse_trajectories]
    x_forward, work_forward = trajectory_frames_to_arrays(forward_frames)
    x_reverse, work_reverse = trajectory_frames_to_arrays(reverse_frames)
    centers, k_values = read_protocol_centers_and_k(segment.forward_path_file)
    n_time = min(
        x_forward.shape[1] if x_forward.ndim == 2 else 0,
        work_forward.shape[1] if work_forward.ndim == 2 else 0,
        x_reverse.shape[1] if x_reverse.ndim == 2 else 0,
        work_reverse.shape[1] if work_reverse.ndim == 2 else 0,
        len(centers),
        len(k_values),
    )
    centers = centers[:n_time]
    k_values = k_values[:n_time]
    left_reference_x = float(segment.left_boundary.mean_x)
    right_reference_x = float(segment.right_boundary.mean_x)
    cft = solve_segment_cft_delta_f_once(
        work_forward[:, :n_time],
        work_reverse[:, :n_time],
        kT=float(ctx["thermal_kT"]),
    )
    fixed_delta_f = (
        float(cft["delta_f"])
        if bool(cft.get("cft_solved", False))
        and cft.get("delta_f") is not None
        and math.isfinite(float(cft["delta_f"]))
        and not bool(disable_fixed_cft_bootstrap)
        else None
    )
    recompute_delta_f_per_bootstrap = bool(
        disable_fixed_cft_bootstrap or fixed_delta_f is None
    )
    left_boot = bootstrap_bidirectional_mts_pmf(
        x_forward,
        work_forward,
        x_reverse,
        work_reverse,
        centers,
        k_values,
        grid,
        reference_x=left_reference_x,
        kT=float(ctx["thermal_kT"]),
        n_boot=int(n_boot),
        fk_boot=max(int(n_boot // 8), 4),
        rng_seed=int(rng_seed),
        fixed_delta_f=fixed_delta_f,
        recompute_delta_f_per_bootstrap=recompute_delta_f_per_bootstrap,
    )
    right_boot = bootstrap_bidirectional_mts_pmf(
        x_forward,
        work_forward,
        x_reverse,
        work_reverse,
        centers,
        k_values,
        grid,
        reference_x=right_reference_x,
        kT=float(ctx["thermal_kT"]),
        n_boot=int(n_boot),
        fk_boot=max(int(n_boot // 8), 4),
        rng_seed=int(rng_seed + 100000),
        fixed_delta_f=fixed_delta_f,
        recompute_delta_f_per_bootstrap=recompute_delta_f_per_bootstrap,
    )
    pmf = shift_finite_to_min_zero(np.asarray(left_boot["pmf_ref0"], dtype=float))
    var_left = np.asarray(left_boot["var_ref0"], dtype=float)
    var_right = np.asarray(right_boot["var_ref0"], dtype=float)
    variance = np.fmin(var_left, var_right)
    coverage = np.isfinite(pmf) & np.isfinite(variance)
    neq_x_parts = []
    if x_forward.ndim == 2:
        neq_x_parts.append(x_forward.ravel())
    if x_reverse.ndim == 2:
        neq_x_parts.append(x_reverse.ravel())
    if neq_x_parts:
        combined_neq_x = np.concatenate(neq_x_parts)
        finite_neq_x = combined_neq_x[np.isfinite(combined_neq_x)]
        if finite_neq_x.size > 0:
            coverage &= coverage_mask_from_samples(finite_neq_x, grid)
    cft_summary = {
        "cft_solved_once": bool(cft.get("cft_solved", False)),
        "cft_delta_f": cft.get("delta_f", None),
        "cft_delta_f_unc": cft.get("delta_f_unc", None),
        "cft_method": cft.get("method", "BAR"),
        "cft_reason": cft.get("reason", ""),
        "bootstrap_recomputed_cft": bool(recompute_delta_f_per_bootstrap),
        "fixed_delta_f_used": fixed_delta_f,
    }
    segment.cft_summary = cft_summary
    if not np.any(coverage):
        segment.mts_patch_built = False
        segment.neq_patch_decision = {
            "segment": segment.name,
            "left_boundary": segment.left_boundary.name,
            "right_boundary": segment.right_boundary.name,
            "eop_crossed": segment.connectivity.get("eop_crossed", ""),
            "eop_jsd_raw": segment.connectivity.get("eop_jsd_raw", ""),
            "eop_jsd_norm": segment.connectivity.get("eop_jsd_norm", ""),
            "eop_common_bins": segment.connectivity.get("eop_common_bins", ""),
            "eop_interval_overlap": segment.connectivity.get("eop_interval_overlap", ""),
            "mts_patch_built": 0,
            "included_in_global_pmf": 0,
            "cft_solved_once": int(bool(cft_summary["cft_solved_once"])),
            "cft_delta_f": cft_summary["cft_delta_f"],
            "cft_delta_f_unc": cft_summary["cft_delta_f_unc"],
            "bootstrap_recomputed_cft": int(bool(cft_summary["bootstrap_recomputed_cft"])),
            "reason": "neq_mts_patch_no_finite_coverage",
        }
        raise RuntimeError(f"{segment.name} produced no finite NEQ_MTS coverage.")
    patch = PMFPatch(
        name=segment.name,
        kind="NEQ_MTS",
        root=patch_dir,
        grid=np.asarray(grid, dtype=float),
        pmf=pmf,
        variance=variance,
        coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[source_name(segment.left), source_name(segment.right)],
        metadata={
            "segment_name": segment.name,
            "left_name": source_name(segment.left),
            "right_name": source_name(segment.right),
            "left_boundary": segment.left_boundary.name,
            "right_boundary": segment.right_boundary.name,
            "anchor_coordinate": "mean_x",
            "left_reference_x": left_reference_x,
            "right_reference_x": right_reference_x,
            "left_reference_x_most": float(segment.left_boundary.x_most),
            "right_reference_x_most": float(segment.right_boundary.x_most),
            "delta_f": float(left_boot["delta_f"]),
            "delta_f_unc": float(left_boot["delta_f_unc"]),
            "n_boot_left": int(left_boot["n_boot_used"]),
            "n_boot_right": int(right_boot["n_boot_used"]),
            **cft_summary,
        },
        anchor_variances={
            "var_left_anchor": var_left,
            "var_right_anchor": var_right,
            "var_neq_min": variance,
        },
    )
    segment.mts_patch_built = bool(np.any(np.isfinite(pmf)) and np.any(np.isfinite(variance)))
    segment.neq_patch_decision = {
        "segment": segment.name,
        "left_boundary": segment.left_boundary.name,
        "right_boundary": segment.right_boundary.name,
        "eop_crossed": segment.connectivity.get("eop_crossed", ""),
        "eop_jsd_raw": segment.connectivity.get("eop_jsd_raw", ""),
        "eop_jsd_norm": segment.connectivity.get("eop_jsd_norm", ""),
        "eop_common_bins": segment.connectivity.get("eop_common_bins", ""),
        "eop_interval_overlap": segment.connectivity.get("eop_interval_overlap", ""),
        "mts_patch_built": int(bool(segment.mts_patch_built)),
        "included_in_global_pmf": int(bool(segment.mts_patch_built)),
        "cft_solved_once": int(bool(cft_summary["cft_solved_once"])),
        "cft_delta_f": cft_summary["cft_delta_f"],
        "cft_delta_f_unc": cft_summary["cft_delta_f_unc"],
        "bootstrap_recomputed_cft": int(bool(cft_summary["bootstrap_recomputed_cft"])),
        "reason": "built_neq_mts_patch",
    }
    segment_summary_path = segment.root / "segment_summary.json"
    segment_summary = load_json(segment_summary_path)
    if segment_summary:
        segment_summary.update(cft_summary)
        segment_summary["mts_patch_built"] = bool(segment.mts_patch_built)
        write_json(segment_summary_path, segment_summary)
    write_patch_outputs(patch, analytic_pmf(grid, ctx))
    return patch


def write_patch_outputs(patch: PMFPatch, analytic: np.ndarray) -> None:
    patch.root.mkdir(parents=True, exist_ok=True)
    pmf_rows: list[dict[str, Any]] = []
    for idx, x_value in enumerate(patch.grid):
        pmf_rows.append(
            {
                "x": float(x_value),
                "local_pmf": float(patch.pmf[idx]) if np.isfinite(patch.pmf[idx]) else "",
                "variance": float(patch.variance[idx]) if np.isfinite(patch.variance[idx]) else "",
                "coverage": int(bool(patch.coverage_mask[idx])),
                "analytic_pmf": float(analytic[idx]) if np.isfinite(analytic[idx]) else "",
            }
        )
    write_csv(
        patch.root / "pmf.csv",
        ["x", "local_pmf", "variance", "coverage", "analytic_pmf"],
        pmf_rows,
    )
    variance_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    anchor_fieldnames = ["x", *list(patch.anchor_variances.keys())]
    for idx, x_value in enumerate(patch.grid):
        variance_rows.append(
            {
                "x": float(x_value),
                "variance": float(patch.variance[idx]) if np.isfinite(patch.variance[idx]) else "",
                "coverage": int(bool(patch.coverage_mask[idx])),
            }
        )
        anchor_row: dict[str, Any] = {"x": float(x_value)}
        for anchor_name, anchor_values in patch.anchor_variances.items():
            anchor_row[anchor_name] = (
                float(anchor_values[idx]) if np.isfinite(anchor_values[idx]) else ""
            )
        anchor_rows.append(anchor_row)
    write_csv(
        patch.root / "bootstrap_variance.csv",
        ["x", "variance", "coverage"],
        variance_rows,
    )
    write_csv(
        patch.root / "anchor_variances.csv",
        anchor_fieldnames,
        anchor_rows,
    )
    write_json(
        patch.root / "patch_summary.json",
        {
            "name": patch.name,
            "kind": patch.kind,
            "source_names": patch.source_names,
            "n_covered_bins": int(np.count_nonzero(patch.coverage_mask)),
            "metadata": patch.metadata,
        },
    )


def write_patch_offset_aligned_outputs(
    patches: list[PMFPatch],
    fit_details: dict[str, Any],
) -> None:
    patch_offsets = fit_details.get("patch_offsets", {})
    for patch in patches:
        patch_offset = patch_offsets.get(patch.name, None)
        rows: list[dict[str, Any]] = []
        for idx, x_value in enumerate(patch.grid):
            local_pmf = patch.pmf[idx]
            variance = patch.variance[idx]
            coverage = bool(patch.coverage_mask[idx])
            aligned_pmf = (
                float(local_pmf + patch_offset)
                if coverage and patch_offset is not None and np.isfinite(local_pmf)
                else ""
            )
            rows.append(
                {
                    "x": float(x_value),
                    "local_pmf": float(local_pmf) if np.isfinite(local_pmf) else "",
                    "patch_offset": patch_offset if patch_offset is not None else "",
                    "aligned_pmf": aligned_pmf,
                    "variance": float(variance) if np.isfinite(variance) else "",
                    "coverage": int(coverage),
                }
            )
        write_csv(
            patch.root / "pmf_offset_aligned.csv",
            ["x", "local_pmf", "patch_offset", "aligned_pmf", "variance", "coverage"],
            rows,
        )


def fit_global_pmf_from_patches(
    patches: list[PMFPatch],
    grid: np.ndarray,
    variance_floor: float,
    reference_x: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not patches:
        nan = np.full(len(grid), np.nan, dtype=float)
        return nan.copy(), nan.copy(), {
            "patch_offsets": {},
            "n_patches": 0,
            "n_observations": 0,
            "gauge_reference_x": None,
            "rms_residual": None,
        }
    observation_rows = []
    for patch_idx, patch in enumerate(patches):
        finite = np.asarray(patch.coverage_mask, dtype=bool)
        finite &= np.isfinite(patch.pmf) & np.isfinite(patch.variance)
        for grid_idx in np.where(finite)[0].tolist():
            weight = 1.0 / (float(patch.variance[grid_idx]) + float(variance_floor))
            observation_rows.append((patch_idx, grid_idx, float(patch.pmf[grid_idx]), weight))
    if not observation_rows:
        nan = np.full(len(grid), np.nan, dtype=float)
        return nan.copy(), nan.copy(), {
            "patch_offsets": {patch.name: None for patch in patches},
            "n_patches": len(patches),
            "n_observations": 0,
            "gauge_reference_x": None,
            "rms_residual": None,
        }

    n_grid = len(grid)
    n_patch = len(patches)
    gauge_idx = (
        int(np.argmin(np.abs(np.asarray(grid, dtype=float) - float(reference_x))))
        if reference_x is not None
        else int(observation_rows[0][1])
    )
    n_rows = len(observation_rows) + 1
    a = np.zeros((n_rows, n_grid + n_patch), dtype=float)
    b = np.zeros(n_rows, dtype=float)
    for row_idx, (patch_idx, grid_idx, pmf_value, weight) in enumerate(observation_rows):
        scale = math.sqrt(weight)
        a[row_idx, grid_idx] = scale
        a[row_idx, n_grid + patch_idx] = -scale
        b[row_idx] = scale * pmf_value
    a[-1, gauge_idx] = 1.0e6
    b[-1] = 0.0
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    global_pmf = np.asarray(solution[:n_grid], dtype=float)
    patch_offsets = np.asarray(solution[n_grid:], dtype=float)
    if reference_x is None:
        finite_global = np.isfinite(global_pmf)
        shift = float(np.nanmin(global_pmf[finite_global])) if np.any(finite_global) else 0.0
        global_pmf[finite_global] -= shift
        patch_offsets -= shift
    global_variance = np.full(n_grid, np.nan, dtype=float)
    n_cover = np.zeros(n_grid, dtype=int)
    for grid_idx in range(n_grid):
        weights = []
        for patch in patches:
            if (
                bool(patch.coverage_mask[grid_idx])
                and np.isfinite(patch.variance[grid_idx])
                and np.isfinite(patch.pmf[grid_idx])
            ):
                weights.append(1.0 / (float(patch.variance[grid_idx]) + float(variance_floor)))
        if weights:
            global_variance[grid_idx] = 1.0 / float(np.sum(weights))
            n_cover[grid_idx] = len(weights)
    global_pmf[n_cover <= 0] = np.nan
    residuals = []
    for patch_idx, grid_idx, pmf_value, weight in observation_rows:
        residual = float(global_pmf[grid_idx] - pmf_value - patch_offsets[patch_idx])
        residuals.append(weight * residual * residual)
    details = {
        "patch_offsets": {
            patch.name: float(patch_offsets[idx]) for idx, patch in enumerate(patches)
        },
        "n_patches": int(len(patches)),
        "n_observations": int(len(observation_rows)),
        "gauge_reference_x": float(grid[gauge_idx]),
        "reference_x_argument": None if reference_x is None else float(reference_x),
        "rms_residual": float(math.sqrt(np.mean(residuals))) if residuals else None,
        "n_covering_patches": n_cover.tolist(),
    }
    return global_pmf, global_variance, details


def endpoint_x_from_trajectories(trajectories: list[list[dict[str, str]]]) -> np.ndarray:
    values = [float(rows[-1]["x"]) for rows in trajectories if rows and rows[-1].get("x", "") != ""]
    return np.asarray(values, dtype=float)


def neq_eop_quantile_crossing(
    segment: "NEQSegment",
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[str, Any]:
    fwd = endpoint_x_from_trajectories(segment.forward_trajectories)
    rev = endpoint_x_from_trajectories(segment.reverse_trajectories)
    fwd = fwd[np.isfinite(fwd)]
    rev = rev[np.isfinite(rev)]
    if fwd.size < 2 or rev.size < 2:
        return {
            "crossed": False,
            "fwd_q_high": float("nan"),
            "rev_q_low": float("nan"),
            "q_low": float(q_low),
            "q_high": float(q_high),
            "reason": "not_enough_samples",
        }
    fwd_hi = float(np.quantile(fwd, q_high))
    rev_lo = float(np.quantile(rev, q_low))
    return {
        "crossed": bool(fwd_hi >= rev_lo),
        "fwd_q_high": fwd_hi,
        "rev_q_low": rev_lo,
        "q_low": float(q_low),
        "q_high": float(q_high),
        "reason": "",
    }


def eq_tail_quantile_crossing(
    left_window: "EnsembleWindow",
    right_window: "EnsembleWindow",
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[str, Any]:
    left_x = eq_tail_samples(left_window)
    right_x = eq_tail_samples(right_window)
    left_x = left_x[np.isfinite(left_x)]
    right_x = right_x[np.isfinite(right_x)]
    if left_x.size < 2 or right_x.size < 2:
        return {
            "crossed": False,
            "left_q_high": float("nan"),
            "right_q_low": float("nan"),
            "q_low": float(q_low),
            "q_high": float(q_high),
            "reason": "not_enough_samples",
        }
    left_hi = float(np.quantile(left_x, q_high))
    right_lo = float(np.quantile(right_x, q_low))
    return {
        "crossed": bool(left_hi >= right_lo),
        "left_q_high": left_hi,
        "right_q_low": right_lo,
        "q_low": float(q_low),
        "q_high": float(q_high),
        "reason": "",
    }


def ensure_segment_connectivity(
    segment: NEQSegment,
    grid: np.ndarray,
    js_threshold_norm: float,
    out_root: Path,
    mts_patch_built: bool | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    if not segment.connectivity:
        segment.connectivity = eop_distributions_cross(
            endpoint_x_from_trajectories(segment.forward_trajectories),
            endpoint_x_from_trajectories(segment.reverse_trajectories),
            grid,
            js_threshold_norm,
        )
    if mts_patch_built is not None:
        segment.mts_patch_built = bool(mts_patch_built)
    summary = {
        **segment.connectivity,
        "segment": segment.name,
        "left_boundary": segment.left_boundary.name,
        "right_boundary": segment.right_boundary.name,
        "mts_patch_built": bool(segment.mts_patch_built),
        "reason": (
            reason
            if reason is not None
            else ("built_neq_mts_patch" if segment.mts_patch_built else "eop_disconnected_skip_mts")
        ),
        "connectivity_summary_file": relative_to_root(segment.root / "connectivity_summary.json", out_root),
    }
    segment.connectivity = summary
    write_json(segment.root / "connectivity_summary.json", summary)
    return summary


def trajectory_log_weight_and_bin(
    rows: list[dict[str, str]],
    grid: np.ndarray,
    grid_dx: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambdas = []
    log_weights = []
    x_indices = []
    for row in rows:
        lambdas.append(float(row.get("lambda", "0.0") or 0.0))
        work_value = float(row.get("work", "nan") or "nan")
        log_weights.append(-beta * work_value if math.isfinite(work_value) else float("nan"))
        x_value = float(row.get("x", "nan") or "nan")
        if math.isfinite(x_value):
            x_idx = int(np.argmin(np.abs(np.asarray(grid, dtype=float) - x_value)))
            if abs(float(grid[x_idx]) - x_value) <= max(0.5 * grid_dx, 1.0e-12):
                x_indices.append(x_idx)
            else:
                x_indices.append(-1)
        else:
            x_indices.append(-1)
    return (
        np.asarray(lambdas, dtype=float),
        np.asarray(log_weights, dtype=float),
        np.asarray(x_indices, dtype=int),
    )


def hs_pmf_from_trajectories(
    trajectories: list[list[dict[str, str]]],
    centers: np.ndarray,
    k_values: np.ndarray,
    grid: np.ndarray,
    ctx: dict[str, Any],
) -> np.ndarray:
    if not trajectories:
        return np.full(len(grid), np.nan, dtype=float)
    beta = 1.0 / float(ctx["thermal_kT"])
    grid_dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    n_time = min(len(trajectories[0]), len(centers), len(k_values))
    if n_time <= 0:
        return np.full(len(grid), np.nan, dtype=float)
    log_sum_w = np.full(n_time, -np.inf, dtype=float)
    log_sum_hist = np.full((n_time, len(grid)), -np.inf, dtype=float)
    for rows in trajectories:
        if not rows:
            continue
        lambdas, log_weights, x_indices = trajectory_log_weight_and_bin(rows[:n_time], grid, grid_dx, beta)
        _ = lambdas
        for time_idx in range(min(n_time, len(log_weights), len(x_indices))):
            log_weight = float(log_weights[time_idx])
            if not math.isfinite(log_weight):
                continue
            log_sum_w[time_idx] = np.logaddexp(log_sum_w[time_idx], log_weight)
            x_idx = int(x_indices[time_idx])
            if x_idx >= 0:
                log_sum_hist[time_idx, x_idx] = np.logaddexp(log_sum_hist[time_idx, x_idx], log_weight)
    log_numerator_terms = log_sum_hist - log_sum_w[:, None]
    log_denominator_terms = np.full((n_time, len(grid)), -np.inf, dtype=float)
    log_n_traj = math.log(float(len(trajectories)))
    grid_arr = np.asarray(grid, dtype=float)
    for time_idx in range(n_time):
        if not math.isfinite(float(log_sum_w[time_idx])):
            continue
        center = float(centers[time_idx])
        k_value = float(k_values[time_idx])
        log_denominator_terms[time_idx] = (
            log_n_traj
            - float(log_sum_w[time_idx])
            - beta * 0.5 * k_value * (grid_arr - center) * (grid_arr - center)
        )
    log_numerator = np.asarray(logsumexp_np(log_numerator_terms, axis=0), dtype=float)
    log_denominator = np.asarray(logsumexp_np(log_denominator_terms, axis=0), dtype=float)
    log_density = log_numerator - log_denominator
    finite = np.isfinite(log_density)
    if np.any(finite):
        log_norm = float(logsumexp_np(log_density[finite])) + math.log(grid_dx)
        log_density[finite] -= log_norm
    pmf = np.full(len(grid), np.nan, dtype=float)
    pmf[finite] = -float(ctx["thermal_kT"]) * log_density[finite]
    return shift_finite_to_min_zero(pmf)


def align_pmf_to_reference_x(pmf: np.ndarray, grid: np.ndarray, reference_x: float) -> np.ndarray:
    aligned = np.asarray(pmf, dtype=float).copy()
    if aligned.size <= 0:
        return aligned
    ref_idx = int(np.argmin(np.abs(np.asarray(grid, dtype=float) - float(reference_x))))
    ref_value = float(aligned[ref_idx])
    if math.isfinite(ref_value):
        finite = np.isfinite(aligned)
        aligned[finite] -= ref_value
    return aligned


def bootstrap_hs_patch(
    *,
    segment: NEQSegment,
    direction: str,
    grid: np.ndarray,
    ctx: dict[str, Any],
    n_boot: int,
    rng_seed: int,
    out_root: Path,
) -> tuple[PMFPatch | None, dict[str, Any] | None]:
    if direction == "forward":
        trajectories = segment.forward_trajectories
        centers, k_values = read_protocol_centers_and_k(segment.forward_path_file)
        reference_x = float(segment.left_boundary.mean_x)
        kind = "HS_FWD_FALLBACK"
        source_boundary = segment.left_boundary.name
        target_boundary = segment.right_boundary.name
        output_dir = segment.root / "hs_fallback" / "forward"
    else:
        trajectories = segment.reverse_trajectories
        centers, k_values = read_protocol_centers_and_k(segment.reverse_path_file)
        reference_x = float(segment.right_boundary.mean_x)
        kind = "HS_REV_FALLBACK"
        source_boundary = segment.right_boundary.name
        target_boundary = segment.left_boundary.name
        output_dir = segment.root / "hs_fallback" / "reverse"
    if not trajectories:
        return None, None
    n_time = min(len(centers), len(k_values), *(len(rows) for rows in trajectories if rows))
    if n_time <= 0:
        return None, None
    centers = np.asarray(centers[:n_time], dtype=float)
    k_values = np.asarray(k_values[:n_time], dtype=float)
    trimmed = [rows[:n_time] for rows in trajectories if rows]
    base_pmf = align_pmf_to_reference_x(
        hs_pmf_from_trajectories(trimmed, centers, k_values, grid, ctx),
        grid,
        reference_x,
    )
    if not np.isfinite(base_pmf).any():
        return None, None
    rng = np.random.default_rng(int(rng_seed))
    boot_stack: list[np.ndarray] = []
    for _boot_idx in range(int(n_boot)):
        draw_indices = rng.integers(0, len(trimmed), size=len(trimmed))
        sample = [trimmed[int(idx)] for idx in draw_indices.tolist()]
        boot_pmf = align_pmf_to_reference_x(
            hs_pmf_from_trajectories(sample, centers, k_values, grid, ctx),
            grid,
            reference_x,
        )
        if np.isfinite(boot_pmf).any():
            boot_stack.append(boot_pmf)
    if len(boot_stack) >= 2:
        variance = np.nanvar(np.vstack(boot_stack), axis=0, ddof=1)
        n_boot_used = len(boot_stack)
    elif len(boot_stack) == 1:
        variance = np.full(len(grid), np.nan, dtype=float)
        n_boot_used = 1
    else:
        variance = np.full(len(grid), np.nan, dtype=float)
        n_boot_used = 0
    coverage = np.isfinite(base_pmf)
    patch = PMFPatch(
        name=f"{segment.name}__{direction.upper()}",
        kind=kind,
        root=output_dir,
        grid=np.asarray(grid, dtype=float),
        pmf=base_pmf,
        variance=np.asarray(variance, dtype=float),
        coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[segment.name, direction],
        metadata={
            "segment_name": segment.name,
            "direction": direction,
            "source_boundary": source_boundary,
            "target_boundary": target_boundary,
            "reference_x": reference_x,
            "n_boot": int(n_boot),
            "n_boot_used": int(n_boot_used),
        },
        anchor_variances={"var_hs": np.asarray(variance, dtype=float)},
    )
    write_patch_outputs(patch, analytic_pmf(grid, ctx))
    summary_row = {
        "segment": segment.name,
        "direction": direction,
        "patch_name": patch.name,
        "source_boundary": source_boundary,
        "target_boundary": target_boundary,
        "n_boot": int(n_boot),
        "n_boot_used": int(n_boot_used),
        "n_covered_bins": int(np.count_nonzero(patch.coverage_mask)),
        "reason": "budget_exhausted_disconnected_segment",
        "pmf_file": relative_to_root(output_dir / "pmf.csv", out_root),
        "variance_file": relative_to_root(output_dir / "bootstrap_variance.csv", out_root),
    }
    return patch, summary_row


def build_hs_fallback_patches(
    *,
    disconnected_segments: list[NEQSegment],
    grid: np.ndarray,
    ctx: dict[str, Any],
    n_boot: int,
    base_seed: int,
    out_root: Path,
) -> tuple[list[PMFPatch], list[dict[str, Any]]]:
    patches: list[PMFPatch] = []
    rows: list[dict[str, Any]] = []
    for segment_idx, segment in enumerate(disconnected_segments):
        segment_summary_rows: list[dict[str, Any]] = []
        for direction_idx, direction in enumerate(("forward", "reverse")):
            patch, summary_row = bootstrap_hs_patch(
                segment=segment,
                direction=direction,
                grid=grid,
                ctx=ctx,
                n_boot=n_boot,
                rng_seed=int(base_seed + 900000 + 1000 * segment_idx + 100 * direction_idx),
                out_root=out_root,
            )
            if patch is not None and summary_row is not None:
                patches.append(patch)
                rows.append(summary_row)
                segment_summary_rows.append(summary_row)
        write_json(
            segment.root / "hs_fallback" / "hs_fallback_summary.json",
            {
                "segment": segment.name,
                "left_boundary": segment.left_boundary.name,
                "right_boundary": segment.right_boundary.name,
                "patches": segment_summary_rows,
            },
        )
        segment.hs_fallback_rows = segment_summary_rows
    return patches, rows


def finalize_child_proposal(
    *,
    side: str,
    parent: EnsembleWindow,
    opposite: EnsembleWindow,
    target_source: str,
    q50: float,
    q_anchor: float,
    target_x: float,
    center_raw: float,
    grid: np.ndarray,
    alpha: float,
    k_min: float,
    k_max: float,
) -> tuple[float, float, dict[str, Any]]:
    dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 0.1
    feasible_low = float(parent.center_x + dx) if side == "left" else float(opposite.center_x + dx)
    feasible_high = float(opposite.center_x - dx) if side == "left" else float(parent.center_x - dx)
    progress_possible = bool(feasible_low <= feasible_high)
    if progress_possible:
        center_candidate = min(max(float(center_raw), feasible_low), feasible_high)
    else:
        center_candidate = 0.5 * (float(parent.center_x) + float(opposite.center_x))
    center_x = float(center_candidate)
    if side == "left":
        matched_force = max(float(opposite.k) * (float(opposite.center_x) - float(target_x)), 1.0e-8)
        gap = max(float(center_x) - float(target_x), dx)
    else:
        matched_force = max(float(opposite.k) * (float(target_x) - float(opposite.center_x)), 1.0e-8)
        gap = max(float(target_x) - float(center_x), dx)
    raw_k = float(matched_force / gap)
    parent_tail = eq_tail_samples(parent)
    parent_sigma = float(np.std(parent_tail)) if parent_tail.size > 1 else 0.0
    barrier_crossing_tol = max(0.5 * dx, 0.1 * parent_sigma)
    if side == "left":
        barrier_crossing_displacement = float(parent.x_most) - float(parent.center_x)
    else:
        barrier_crossing_displacement = float(parent.center_x) - float(parent.x_most)
    barrier_crossing = bool(barrier_crossing_displacement > barrier_crossing_tol)

    k_value = float(min(max(raw_k, k_min), k_max))
    k_rule = "force_matching_clipped"
    if raw_k < float(k_min):
        k_clamped_to = "k_min"
    elif raw_k > float(k_max):
        k_clamped_to = "k_max"
    else:
        k_clamped_to = "none"
    barrier_crossing_action = "none_gt_slope_aware"
    return float(center_x), k_value, {
        "side": side,
        "parent": parent.name,
        "opposite": opposite.name,
        "target_source": target_source,
        "endpoint_q50": float(q50),
        "endpoint_q_anchor": float(q_anchor),
        "target_x": float(target_x),
        "center_raw": float(center_raw),
        "center_x": float(center_x),
        "matched_force": float(matched_force),
        "gap": float(gap),
        "raw_k": float(raw_k),
        "k": float(k_value),
        "k_clamped_to": k_clamped_to,
        "barrier_crossing_diagnostic": bool(barrier_crossing),
        "barrier_crossing_action": barrier_crossing_action,
        "barrier_crossing_displacement": float(barrier_crossing_displacement),
        "barrier_crossing_tol": float(barrier_crossing_tol),
        "parent_x_most": float(parent.x_most),
        "parent_center_x": float(parent.center_x),
        "parent_sigma": float(parent_sigma),
        "k_rule": str(k_rule),
        "alpha": float(alpha),
        "progress_possible": progress_possible,
    }


def propose_child_from_neq_endpoints(
    *,
    side: str,
    parent: EnsembleWindow,
    opposite: EnsembleWindow,
    endpoint_x: np.ndarray,
    grid: np.ndarray,
    q_next: float,
    alpha: float,
    x_leap: float,
    k_min: float,
    k_max: float,
    method: str = "leap-fixed",
    k_method: str = "force-matching",
) -> tuple[float, float, dict[str, Any]]:
    finite_endpoints = np.asarray(endpoint_x, dtype=float)
    finite_endpoints = finite_endpoints[np.isfinite(finite_endpoints)]
    if finite_endpoints.size <= 0:
        parent_tail = eq_tail_samples(parent)
        if parent_tail.size <= 0:
            raise RuntimeError(f"Missing samples for child proposal from {parent.name}.")
        q50 = float(np.quantile(parent_tail, 0.5))
        q_anchor = float(np.quantile(parent_tail, q_next if side == "left" else 1.0 - q_next))
        target_x = q_anchor
        center_raw = float(target_x + x_leap) if side == "left" else float(target_x - x_leap)
        center_x, k_value, plan = finalize_child_proposal(
            side=side,
            parent=parent,
            opposite=opposite,
            target_source="fallback_parent_eq_tail",
            q50=q50,
            q_anchor=q_anchor,
            target_x=target_x,
            center_raw=center_raw,
            grid=grid,
            alpha=alpha,
            k_min=k_min,
            k_max=k_max,
        )
        plan.update(
            {
                "endpoint_count": 0,
                "method": method,
                "k_method": k_method,
            }
        )
        return center_x, k_value, plan

    q50 = float(np.quantile(finite_endpoints, 0.5))
    q_anchor = float(np.quantile(finite_endpoints, q_next if side == "left" else 1.0 - q_next))
    target_x = q_anchor
    center_raw = float(target_x + x_leap) if side == "left" else float(target_x - x_leap)
    center_x, k_value, plan = finalize_child_proposal(
        side=side,
        parent=parent,
        opposite=opposite,
        target_source="neq_endpoint_distribution",
        q50=q50,
        q_anchor=q_anchor,
        target_x=target_x,
        center_raw=center_raw,
        grid=grid,
        alpha=alpha,
        k_min=k_min,
        k_max=k_max,
    )
    plan.update(
        {
            "endpoint_count": int(finite_endpoints.size),
            "method": method,
            "k_method": k_method,
        }
    )
    return center_x, k_value, plan


def build_cluster_rows(clusters: list[EQCluster]) -> list[dict[str, Any]]:
    rows = []
    for cluster in clusters:
        rows.append(
            {
                "name": cluster.name,
                "left_x": float(cluster.left_x),
                "right_x": float(cluster.right_x),
                "window_names": ",".join(window.name for window in cluster.windows),
                "n_windows": int(len(cluster.windows)),
                "order_coordinate": "mean_x",
                "left_x_coordinate": "mean_x",
                "right_x_coordinate": "mean_x",
                "window_mean_xs": ",".join(str(float(w.mean_x)) for w in cluster.windows),
                "window_center_xs": ",".join(str(float(w.center_x)) for w in cluster.windows),
            }
        )
    return rows


def build_eq_bar_edge_rows(
    clusters: list[EQCluster],
    js_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    w2c: dict[str, str] = {}
    for cluster in clusters:
        for w in cluster.windows:
            w2c[w.name] = cluster.name
    rows = []
    for row in js_rows:
        if not bool(row.get("merged", False)):
            continue
        ln = str(row.get("left_window", ""))
        rn = str(row.get("right_window", ""))
        cluster_name = w2c.get(ln) or w2c.get(rn) or ""
        rows.append(
            {
                "cluster": cluster_name,
                "left_window": ln,
                "right_window": rn,
                "left_mean_x": row.get("left_mean_x", ""),
                "right_mean_x": row.get("right_mean_x", ""),
                "left_center_x": row.get("left_center_x", ""),
                "right_center_x": row.get("right_center_x", ""),
                "ddF": row.get("bar_delta_f_unc", ""),
                "bar_delta_f": row.get("bar_delta_f", ""),
                "bar_delta_f_unc": row.get("bar_delta_f_unc", ""),
                "bar_solved": row.get("bar_solved", False),
                "bar_method": row.get("bar_method", ""),
                "bar_reason": row.get("bar_reason", ""),
                "pair_jsd": row.get("pair_jsd", ""),
                "pair_jsd_norm": row.get("pair_jsd_norm", ""),
                "merged": True,
                "kind": "EQ_BAR",
                "style": "solid",
                "color": "black",
            }
        )
    return rows


def build_eq_map_segments(windows: list["EnsembleWindow"]) -> list[dict[str, Any]]:
    ordered = sorted(windows, key=lambda w: (float(w.mean_x), str(w.name)))
    rows = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        harmonic = estimate_mean_only_k0_x0_from_eq_pair(left, right)
        k0 = float(harmonic["k0_segment"])
        x0 = float(harmonic["x0_segment"])
        lo = min(float(left.mean_x), float(right.mean_x))
        hi = max(float(left.mean_x), float(right.mean_x))
        is_transition = (
            bool(harmonic["valid"])
            and not bool(harmonic["fallback_used"])
            and math.isfinite(k0)
            and math.isfinite(x0)
            and k0 < 0.0
            and lo <= x0 <= hi
        )
        seg_name = f"{left.name}__{right.name}"
        rows.append({
            "segment_name": seg_name,
            "left_window": left.name,
            "right_window": right.name,
            "left_mean_x": float(left.mean_x),
            "right_mean_x": float(right.mean_x),
            "left_center_x": float(left.center_x),
            "right_center_x": float(right.center_x),
            "left_k": float(left.k),
            "right_k": float(right.k),
            "k0_segment": k0,
            "x0_segment": x0,
            "harmonic_valid": bool(harmonic["valid"]),
            "fallback_used": bool(harmonic["fallback_used"]),
            "fallback_reason": str(harmonic["fallback_reason"]),
            "segment_type": "transition" if is_transition else "regular",
            "classification_reason": (
                "k0<0 and x0 in [m_L,m_R]" if is_transition else harmonic["fallback_reason"] or "k0>=0 or x0 outside means"
            ),
        })
    return rows


def build_segment_rows(
    segments: list[NEQSegment],
    out_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        rows.append(
            {
                "name": segment.name,
                "left_name": source_name(segment.left),
                "right_name": source_name(segment.right),
                "boundary_left": segment.left_boundary.name,
                "boundary_right": segment.right_boundary.name,
                "root": relative_to_root(segment.root, out_root),
                "n_forward": int(len(segment.forward_trajectories)),
                "n_reverse": int(len(segment.reverse_trajectories)),
                "n_neq_traj_requested": int(segment.n_neq_traj_requested),
                "n_neq_traj_actual": int(segment.n_neq_traj_actual),
                "neq_budget_limited": int(bool(segment.neq_budget_limited)),
                "neq_cost_requested": int(segment.neq_cost_requested),
                "neq_cost_actual": int(segment.neq_cost_actual),
                "remaining_budget_before_segment": int(segment.remaining_budget_before_segment),
                "forward_path_file": relative_to_root(segment.forward_path_file, out_root),
                "reverse_path_file": relative_to_root(segment.reverse_path_file, out_root),
                "forward_dir": relative_to_root(segment.root / "forward", out_root),
                "reverse_dir": relative_to_root(segment.root / "reverse", out_root),
                "forward_endpoints_file": relative_to_root(segment.root / "forward" / "forward_endpoints.csv", out_root),
                "reverse_endpoints_file": relative_to_root(segment.root / "reverse" / "reverse_endpoints.csv", out_root),
                "connectivity_summary_file": relative_to_root(segment.root / "connectivity_summary.json", out_root),
                "protocol_mode": segment.protocol_mode,
                "protocol_k_mean": segment.protocol_k,
                "protocol_k_min_observed": segment.protocol_k_min,
                "protocol_k_max_observed": segment.protocol_k_max,
                "protocol_x_min_observed": segment.protocol_x_min,
                "protocol_x_max_observed": segment.protocol_x_max,
                "protocol_clip_fraction_k": segment.protocol_clip_fraction_k,
                "protocol_clip_fraction_x": segment.protocol_clip_fraction_x,
                "protocol_summary_file": relative_to_root(segment.root / "protocols" / "protocol_summary.json", out_root),
                "forward_protocol_diagnostics_file": relative_to_root(segment.root / "protocols" / "forward_protocol_diagnostics.csv", out_root),
                "reverse_protocol_diagnostics_file": relative_to_root(segment.root / "protocols" / "reverse_protocol_diagnostics.csv", out_root),
                "eop_crossed": segment.connectivity.get("eop_crossed", ""),
                "eop_jsd_raw": segment.connectivity.get("eop_jsd_raw", ""),
                "eop_jsd_norm": segment.connectivity.get("eop_jsd_norm", ""),
                "eop_jsd_threshold": segment.connectivity.get("eop_jsd_threshold", ""),
                "eop_common_bins": segment.connectivity.get("eop_common_bins", ""),
                "eop_interval_overlap": segment.connectivity.get("eop_interval_overlap", ""),
                "mts_patch_built": int(bool(segment.mts_patch_built)),
                "reason": segment.connectivity.get("reason", ""),
                "cft_solved_once": int(bool(segment.cft_summary.get("cft_solved_once", False))),
                "cft_delta_f": segment.cft_summary.get("cft_delta_f", ""),
                "cft_delta_f_unc": segment.cft_summary.get("cft_delta_f_unc", ""),
                "bootstrap_recomputed_cft": int(bool(segment.cft_summary.get("bootstrap_recomputed_cft", False))),
            }
        )
    return rows


def build_disconnected_segment_rows(
    segments: list[NEQSegment],
    out_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        if bool(segment.connectivity.get("eop_crossed", True)):
            continue
        rows.append(
            {
                "segment": segment.name,
                "left_boundary": segment.left_boundary.name,
                "right_boundary": segment.right_boundary.name,
                "eop_jsd_raw": segment.connectivity.get("eop_jsd_raw", ""),
                "eop_jsd_norm": segment.connectivity.get("eop_jsd_norm", ""),
                "eop_jsd_threshold": segment.connectivity.get("eop_jsd_threshold", ""),
                "eop_common_bins": segment.connectivity.get("eop_common_bins", ""),
                "eop_interval_overlap": segment.connectivity.get("eop_interval_overlap", ""),
                "reason": segment.connectivity.get("reason", "eop_disconnected_skip_mts"),
                "connectivity_summary_file": relative_to_root(segment.root / "connectivity_summary.json", out_root),
            }
        )
    return rows


def build_neq_patch_decision_rows(segments: list[NEQSegment]) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        if not segment.neq_patch_decision and not segment.mts_patch_built and not segment.cft_summary:
            continue
        rows.append(
            {
                "segment": segment.name,
                "left_boundary": segment.left_boundary.name,
                "right_boundary": segment.right_boundary.name,
                "eop_crossed": segment.connectivity.get("eop_crossed", ""),
                "eop_jsd_raw": segment.connectivity.get("eop_jsd_raw", ""),
                "eop_jsd_norm": segment.connectivity.get("eop_jsd_norm", ""),
                "eop_common_bins": segment.connectivity.get("eop_common_bins", ""),
                "eop_interval_overlap": segment.connectivity.get("eop_interval_overlap", ""),
                "mts_patch_built": int(bool(segment.mts_patch_built)),
                "included_in_global_pmf": int(bool(segment.mts_patch_built)),
                "cft_solved_once": int(bool(segment.cft_summary.get("cft_solved_once", False))),
                "cft_delta_f": segment.cft_summary.get("cft_delta_f", ""),
                "cft_delta_f_unc": segment.cft_summary.get("cft_delta_f_unc", ""),
                "bootstrap_recomputed_cft": int(bool(segment.cft_summary.get("bootstrap_recomputed_cft", False))),
                "reason": (
                    segment.neq_patch_decision.get("reason", "")
                    if segment.neq_patch_decision
                    else segment.connectivity.get("reason", "")
                ),
            }
        )
    return rows


def build_mts_failed_rows(
    segments: list[NEQSegment],
    out_root: Path,
) -> list[dict[str, Any]]:
    rows = []
    for segment in segments:
        if not segment.neq_patch_decision and not segment.mts_patch_built and not segment.cft_summary:
            continue
        if bool(segment.mts_patch_built):
            continue
        row = (
            segment.neq_patch_decision
            if segment.neq_patch_decision
            else {
                "segment": segment.name,
                "left_boundary": segment.left_boundary.name,
                "right_boundary": segment.right_boundary.name,
                "eop_crossed": segment.connectivity.get("eop_crossed", ""),
                "eop_jsd_raw": segment.connectivity.get("eop_jsd_raw", ""),
                "eop_jsd_norm": segment.connectivity.get("eop_jsd_norm", ""),
                "eop_common_bins": segment.connectivity.get("eop_common_bins", ""),
                "eop_interval_overlap": segment.connectivity.get("eop_interval_overlap", ""),
                "mts_patch_built": 0,
                "included_in_global_pmf": 0,
                "cft_solved_once": int(bool(segment.cft_summary.get("cft_solved_once", False))),
                "cft_delta_f": segment.cft_summary.get("cft_delta_f", ""),
                "cft_delta_f_unc": segment.cft_summary.get("cft_delta_f_unc", ""),
                "bootstrap_recomputed_cft": int(bool(segment.cft_summary.get("bootstrap_recomputed_cft", False))),
                "reason": segment.connectivity.get("reason", ""),
            }
        )
        rows.append(
            {
                **row,
                "connectivity_summary_file": relative_to_root(
                    segment.root / "connectivity_summary.json",
                    out_root,
                ),
            }
        )
    return rows


def build_patch_rows(
    patches: list[PMFPatch],
    fit_details: dict[str, Any],
    out_root: Path,
) -> list[dict[str, Any]]:
    patch_offsets = fit_details.get("patch_offsets", {})
    rows = []
    for patch in patches:
        rows.append(
            {
                "name": patch.name,
                "kind": patch.kind,
                "source_names": ",".join(patch.source_names),
                "n_covered_bins": int(np.count_nonzero(patch.coverage_mask)),
                "offset": patch_offsets.get(patch.name, ""),
                "patch_root": relative_to_root(patch.root, out_root),
                "pmf_file": relative_to_root(patch.root / "pmf.csv", out_root),
                "variance_file": relative_to_root(patch.root / "bootstrap_variance.csv", out_root),
                "anchor_variances_file": relative_to_root(patch.root / "anchor_variances.csv", out_root),
                "aligned_pmf_file": relative_to_root(patch.root / "pmf_offset_aligned.csv", out_root),
                "summary_file": relative_to_root(patch.root / "patch_summary.json", out_root),
            }
        )
    return rows


def _best_variance_patch_info(
    idx: int,
    patches: list["PMFPatch"],
    patch_offsets: dict[str, float],
) -> dict[str, Any]:
    best_name = ""
    best_kind = ""
    best_var = float("nan")
    for patch in patches:
        cov = np.asarray(patch.coverage_mask, dtype=bool)
        if not cov[idx]:
            continue
        v = float(patch.variance[idx]) if np.isfinite(patch.variance[idx]) else float("inf")
        if math.isfinite(v) and (not math.isfinite(best_var) or v < best_var):
            best_var = v
            best_name = str(patch.name)
            best_kind = str(patch.kind)
    return {
        "best_variance_patch": best_name,
        "best_variance_patch_kind": best_kind,
        "best_variance_value": float(best_var) if math.isfinite(best_var) else "",
    }


def write_global_outputs(
    out_root: Path,
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: np.ndarray,
    fit_details: dict[str, Any],
    ctx: dict[str, Any],
    *,
    pmf_filename: str = "global_pmf.csv",
    fit_summary_filename: str = "global_fit_summary.json",
    patches: list["PMFPatch"] | None = None,
) -> None:
    analytic = analytic_pmf(grid, ctx)
    n_covering = fit_details.get("n_covering_patches", [0] * len(grid))
    patch_offsets: dict[str, float] = fit_details.get("patch_offsets", {})
    patches_list: list[Any] = list(patches) if patches is not None else []
    rows = []
    for idx, x_value in enumerate(grid):
        row: dict[str, Any] = {
            "x": float(x_value),
            "global_pmf": float(global_pmf[idx]) if np.isfinite(global_pmf[idx]) else "",
            "global_variance": (
                float(global_variance[idx]) if np.isfinite(global_variance[idx]) else ""
            ),
            "global_std": (
                float(math.sqrt(global_variance[idx]))
                if np.isfinite(global_variance[idx]) and float(global_variance[idx]) >= 0.0
                else ""
            ),
            "n_covering_patches": int(n_covering[idx]),
            "analytic_pmf": float(analytic[idx]) if np.isfinite(analytic[idx]) else "",
        }
        if patches_list:
            bv = _best_variance_patch_info(idx, patches_list, patch_offsets)
            row.update(bv)
            n_eq = sum(
                1 for p in patches_list
                if np.asarray(p.coverage_mask, dtype=bool)[idx]
                and str(p.kind).startswith("EQ")
            )
            n_neq = sum(
                1 for p in patches_list
                if np.asarray(p.coverage_mask, dtype=bool)[idx]
                and str(p.kind).startswith("NEQ")
            )
            row["n_eq_covering_patches"] = n_eq
            row["n_neq_covering_patches"] = n_neq
        rows.append(row)
    fieldnames = [
        "x",
        "global_pmf",
        "global_variance",
        "global_std",
        "n_covering_patches",
        "analytic_pmf",
    ]
    if patches_list:
        fieldnames += [
            "best_variance_patch",
            "best_variance_patch_kind",
            "best_variance_value",
            "n_eq_covering_patches",
            "n_neq_covering_patches",
        ]
    write_csv(out_root / pmf_filename, fieldnames, rows)
    write_json(out_root / fit_summary_filename, fit_details)


def write_generation_side_outputs(
    side_root: Path,
    protocol_source: Path,
    traj_files: list[Path],
    traj_rows: list[list[dict[str, str]]],
    out_root: Path,
    child_design: dict[str, Any],
) -> None:
    protocol_dir = side_root / "protocols"
    forward_base_dir = side_root / "forward_base"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    forward_base_dir.mkdir(parents=True, exist_ok=True)
    copy_file(protocol_source, protocol_dir / "base_forward_path.csv")
    summarize_trajectory_rows(
        traj_files,
        traj_rows,
        forward_base_dir,
        "forward_endpoints.csv",
        out_root,
    )
    write_json(side_root / "child_design.json", child_design)


def write_patch_bin_contributions(
    out_root: Path,
    grid: np.ndarray,
    patches: list["PMFPatch"],
    fit_details: dict[str, Any],
) -> None:
    patch_offsets: dict[str, float] = fit_details.get("patch_offsets", {})
    rows = []
    for patch in patches:
        cov = np.asarray(patch.coverage_mask, dtype=bool)
        for idx, x_val in enumerate(grid):
            covered = bool(cov[idx])
            local_pmf = float(patch.pmf[idx]) if covered and np.isfinite(patch.pmf[idx]) else ""
            variance = float(patch.variance[idx]) if covered and np.isfinite(patch.variance[idx]) else ""
            offset = patch_offsets.get(patch.name, float("nan"))
            aligned_pmf = (
                float(patch.pmf[idx]) + float(offset)
                if covered and np.isfinite(patch.pmf[idx]) and math.isfinite(float(offset))
                else ""
            )
            rows.append({
                "x": float(x_val),
                "patch_name": str(patch.name),
                "patch_kind": str(patch.kind),
                "covered": int(covered),
                "local_pmf": local_pmf,
                "variance": variance,
                "aligned_pmf": aligned_pmf,
                "patch_offset": float(offset) if math.isfinite(float(offset)) else "",
            })
    write_csv(
        out_root / "patch_bin_contributions.csv",
        ["x", "patch_name", "patch_kind", "covered", "local_pmf", "variance", "aligned_pmf", "patch_offset"],
        rows,
    )


def build_window_to_cluster_map(clusters: list[EQCluster]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cluster in clusters:
        for w in cluster.windows:
            mapping[w.name] = cluster.name
    return mapping


def write_all_neq_patches_csv(
    out_root: Path,
    neq_patch_store: dict[str, "PMFPatch"],
    neq_patch_status: dict[str, dict[str, Any]],
    out_root_base: Path,
) -> None:
    rows = []
    for seg_name, patch in neq_patch_store.items():
        status = neq_patch_status.get(seg_name, {})
        rows.append({
            "segment": seg_name,
            "left_boundary": status.get("left_boundary", ""),
            "right_boundary": status.get("right_boundary", ""),
            "patch_name": patch.name,
            "patch_kind": patch.kind,
            "n_covered_bins": status.get("n_covered_bins", int(np.count_nonzero(patch.coverage_mask))),
            "is_current_neighbor_edge": status.get("is_current_neighbor_edge", ""),
            "is_internal_to_current_eq_cluster": status.get("is_internal_to_current_eq_cluster", ""),
            "is_long_range_or_obsolete_edge": status.get("is_long_range_or_obsolete_edge", ""),
            "left_current_cluster": status.get("left_current_cluster", ""),
            "right_current_cluster": status.get("right_current_cluster", ""),
            "mts_patch_built": status.get("mts_patch_built", 1),
            "reused": status.get("reused", 0),
            "included_in_global_fit": status.get("included_in_global_fit", ""),
            "reason": status.get("reason", ""),
            "patch_root": relative_to_root(patch.root, out_root_base),
            "pmf_file": relative_to_root(patch.root / "pmf.csv", out_root_base),
            "variance_file": relative_to_root(patch.root / "bootstrap_variance.csv", out_root_base),
            "summary_file": relative_to_root(patch.root / "patch_summary.json", out_root_base),
        })
    write_csv(
        out_root / "all_neq_patches.csv",
        [
            "segment", "left_boundary", "right_boundary", "patch_name", "patch_kind",
            "n_covered_bins",
            "is_current_neighbor_edge", "is_internal_to_current_eq_cluster",
            "is_long_range_or_obsolete_edge", "left_current_cluster", "right_current_cluster",
            "mts_patch_built", "reused", "included_in_global_fit", "reason",
            "patch_root", "pmf_file", "variance_file", "summary_file",
        ],
        rows,
    )


def write_patches_used_for_global_fit_csv(
    out_root: Path,
    patches_for_global: list["PMFPatch"],
    fit_details: dict[str, Any],
    neq_patch_store: dict[str, "PMFPatch"],
    neq_patch_status: dict[str, dict[str, Any]],
    out_root_base: Path,
) -> None:
    stored_neq_names = set(neq_patch_store.keys())
    patch_offsets: dict[str, float] = fit_details.get("patch_offsets", {})
    rows = []
    for patch in patches_for_global:
        if str(patch.kind) == "EQ_MBAR":
            inclusion_source = "current_eq_cluster_patch"
        elif patch.name in stored_neq_names:
            _st = neq_patch_status.get(patch.name, {})
            if int(_st.get("is_current_neighbor_edge", 0)):
                inclusion_source = "current_neighbor_neq_patch"
            else:
                inclusion_source = "retained_old_neq_patch"
        else:
            inclusion_source = "other"
        rows.append({
            "name": patch.name,
            "kind": patch.kind,
            "source_names": ",".join(patch.source_names),
            "n_covered_bins": int(np.count_nonzero(patch.coverage_mask)),
            "included_in_global_fit": 1,
            "inclusion_source": inclusion_source,
            "patch_offset": patch_offsets.get(patch.name, ""),
            "patch_root": relative_to_root(patch.root, out_root_base),
            "pmf_file": relative_to_root(patch.root / "pmf.csv", out_root_base),
            "variance_file": relative_to_root(patch.root / "bootstrap_variance.csv", out_root_base),
            "aligned_pmf_file": relative_to_root(patch.root / "pmf_offset_aligned.csv", out_root_base),
            "summary_file": relative_to_root(patch.root / "patch_summary.json", out_root_base),
        })
    write_csv(
        out_root / "patches_used_for_global_fit.csv",
        [
            "name", "kind", "source_names", "n_covered_bins",
            "included_in_global_fit", "inclusion_source", "patch_offset",
            "patch_root", "pmf_file", "variance_file", "aligned_pmf_file", "summary_file",
        ],
        rows,
    )


def write_state_tables(
    base_root: Path,
    out_root: Path,
    windows: list[EnsembleWindow],
    clusters: list[EQCluster],
    segments: list[NEQSegment],
    patches: list[PMFPatch],
    fit_details: dict[str, Any],
    js_rows: list[dict[str, Any]],
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: np.ndarray,
    ctx: dict[str, Any],
    *,
    hs_fallback_rows: list[dict[str, Any]] | None = None,
    fallback_global: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None,
) -> None:
    base_root.mkdir(parents=True, exist_ok=True)
    window_rows = [build_window_summary(window, out_root) for window in windows]
    write_csv(
        base_root / "windows.csv",
        ordered_fieldnames(window_rows),
        window_rows,
    )
    cluster_rows = build_cluster_rows(clusters)
    write_csv(
        base_root / "clusters.csv",
        ordered_fieldnames(cluster_rows),
        cluster_rows,
    )
    segment_rows = build_segment_rows(segments, out_root)
    write_csv(
        base_root / "segments.csv",
        ordered_fieldnames(segment_rows),
        segment_rows,
    )
    patch_rows = build_patch_rows(patches, fit_details, out_root)
    write_csv(
        base_root / "patches.csv",
        ordered_fieldnames(patch_rows),
        patch_rows,
    )
    write_csv(
        base_root / "neighbor_jsd.csv",
        ordered_fieldnames(
            js_rows,
            extras=[
                "left_window",
                "right_window",
                "left_mean_x",
                "right_mean_x",
                "left_center_x",
                "right_center_x",
                "pair_jsd_raw",
                "pair_jsd_norm",
                "pair_jsd",
                "js_threshold",
                "merged",
                "cluster_order_coordinate",
                "bar_solved",
                "bar_delta_f",
                "bar_delta_f_unc",
                "bar_method",
                "bar_reason",
            ],
        ),
        js_rows,
    )
    eq_bar_edge_rows = build_eq_bar_edge_rows(clusters, js_rows)
    write_csv(
        base_root / "eq_bar_edges.csv",
        ordered_fieldnames(
            eq_bar_edge_rows,
            extras=[
                "cluster", "left_window", "right_window",
                "left_mean_x", "right_mean_x", "left_center_x", "right_center_x",
                "ddF", "bar_delta_f", "bar_delta_f_unc", "bar_solved",
                "bar_method", "bar_reason", "pair_jsd", "pair_jsd_norm",
                "merged", "kind", "style", "color",
            ],
        ),
        eq_bar_edge_rows,
    )
    eq_map_rows = build_eq_map_segments(windows)
    write_csv(
        base_root / "eq_map_segments.csv",
        ordered_fieldnames(
            eq_map_rows,
            extras=[
                "segment_name", "left_window", "right_window",
                "left_mean_x", "right_mean_x", "left_center_x", "right_center_x",
                "left_k", "right_k", "k0_segment", "x0_segment",
                "harmonic_valid", "fallback_used", "fallback_reason",
                "segment_type", "classification_reason",
            ],
        ),
        eq_map_rows,
    )
    write_global_outputs(base_root, grid, global_pmf, global_variance, fit_details, ctx, patches=patches)
    write_patch_bin_contributions(base_root, grid, patches, fit_details)
    disconnected_rows = build_disconnected_segment_rows(segments, out_root)
    write_csv(
        base_root / "disconnected_segments.csv",
        ordered_fieldnames(
            disconnected_rows,
            extras=[
                "segment",
                "left_boundary",
                "right_boundary",
                "eop_jsd_raw",
                "eop_jsd_norm",
                "eop_jsd_threshold",
                "eop_common_bins",
                "eop_interval_overlap",
                "reason",
            ],
        ),
        disconnected_rows,
    )
    neq_patch_rows = build_neq_patch_decision_rows(segments)
    write_csv(
        base_root / "neq_patch_decisions.csv",
        ordered_fieldnames(
            neq_patch_rows,
            extras=[
                "segment",
                "left_boundary",
                "right_boundary",
                "eop_crossed",
                "eop_jsd_raw",
                "eop_jsd_norm",
                "eop_common_bins",
                "eop_interval_overlap",
                "mts_patch_built",
                "included_in_global_pmf",
                "cft_solved_once",
                "cft_delta_f",
                "cft_delta_f_unc",
                "bootstrap_recomputed_cft",
                "reason",
            ],
        ),
        neq_patch_rows,
    )
    mts_failed_rows = build_mts_failed_rows(segments, out_root)
    write_csv(
        base_root / "mts_failed_segments.csv",
        ordered_fieldnames(
            mts_failed_rows,
            extras=[
                "segment",
                "left_boundary",
                "right_boundary",
                "eop_crossed",
                "eop_jsd_raw",
                "eop_jsd_norm",
                "eop_common_bins",
                "eop_interval_overlap",
                "mts_patch_built",
                "included_in_global_pmf",
                "cft_solved_once",
                "cft_delta_f",
                "cft_delta_f_unc",
                "bootstrap_recomputed_cft",
                "reason",
            ],
        ),
        mts_failed_rows,
    )
    write_csv(
        base_root / "hs_fallback_segments.csv",
        ordered_fieldnames(
            list(hs_fallback_rows or []),
            extras=[
                "segment",
                "direction",
                "patch_name",
                "source_boundary",
                "target_boundary",
                "n_boot",
                "n_boot_used",
                "n_covered_bins",
                "reason",
            ],
        ),
        list(hs_fallback_rows or []),
    )
    if fallback_global is not None:
        fallback_pmf, fallback_variance, fallback_fit_details = fallback_global
        write_global_outputs(
            base_root,
            grid,
            fallback_pmf,
            fallback_variance,
            fallback_fit_details,
            ctx,
            pmf_filename="global_pmf_with_hs_fallback.csv",
            fit_summary_filename="global_fit_summary_with_hs_fallback.json",
        )


def write_state_snapshot(
    snapshot_root: Path,
    out_root: Path,
    windows: list[EnsembleWindow],
    clusters: list[EQCluster],
    segments: list[NEQSegment],
    patches: list[PMFPatch],
    fit_details: dict[str, Any],
    js_rows: list[dict[str, Any]],
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: np.ndarray,
    ctx: dict[str, Any],
    *,
    hs_fallback_rows: list[dict[str, Any]] | None = None,
    fallback_global: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None,
    neq_patch_store: dict[str, PMFPatch] | None = None,
    neq_patch_status: dict[str, dict[str, Any]] | None = None,
    patches_for_global: list[PMFPatch] | None = None,
) -> None:
    write_state_tables(
        snapshot_root,
        out_root,
        windows,
        clusters,
        segments,
        patches,
        fit_details,
        js_rows,
        grid,
        global_pmf,
        global_variance,
        ctx,
        hs_fallback_rows=hs_fallback_rows,
        fallback_global=fallback_global,
    )
    if neq_patch_store is not None and neq_patch_status is not None:
        write_all_neq_patches_csv(snapshot_root, neq_patch_store, neq_patch_status, out_root)
    if patches_for_global is not None and neq_patch_store is not None and neq_patch_status is not None:
        write_patches_used_for_global_fit_csv(snapshot_root, patches_for_global, fit_details, neq_patch_store, neq_patch_status, out_root)


def rightmost_mean_window(cluster: EQCluster) -> EnsembleWindow:
    return max(cluster.windows, key=lambda w: float(w.mean_x))


def leftmost_mean_window(cluster: EQCluster) -> EnsembleWindow:
    return min(cluster.windows, key=lambda w: float(w.mean_x))


def choose_connected_boundary_pair(
    left_cluster: EQCluster,
    right_cluster: EQCluster,
    segment_store: dict[tuple[str, str], "NEQSegment"],
) -> tuple["EnsembleWindow", "EnsembleWindow", dict[str, Any]]:
    right_boundary_default = leftmost_mean_window(right_cluster)
    right_name = right_boundary_default.name
    connected_left_candidates = [
        w for w in left_cluster.windows
        if (w.name, right_name) in segment_store
    ]
    if connected_left_candidates:
        left_boundary = min(
            connected_left_candidates,
            key=lambda w: abs(float(w.mean_x) - float(right_boundary_default.mean_x)),
        )
        right_boundary = right_boundary_default
        boundary_pair_reason = "existing_connected_segment_to_right_mean_boundary"
    else:
        left_boundary = rightmost_mean_window(left_cluster)
        right_boundary = right_boundary_default
        boundary_pair_reason = "mean_coordinate_cluster_boundary_fallback"
    metadata: dict[str, Any] = {
        "chosen_left_window": left_boundary.name,
        "chosen_right_window": right_boundary.name,
        "chosen_left_mean_x": float(left_boundary.mean_x),
        "chosen_right_mean_x": float(right_boundary.mean_x),
        "boundary_pair_reason": boundary_pair_reason,
        "connected_left_candidate_names": [w.name for w in connected_left_candidates],
    }
    return left_boundary, right_boundary, metadata


def build_eq_pmf_with_neq_variance(
    eq_patches: list[PMFPatch],
    neq_patches: list[PMFPatch],
    grid: np.ndarray,
) -> list[PMFPatch]:
    neq_variance_grid = np.full(len(grid), np.nan, dtype=float)
    for neq_patch in neq_patches:
        neq_cov = np.asarray(neq_patch.coverage_mask, dtype=bool) & np.isfinite(neq_patch.variance)
        neq_variance_grid = np.where(
            neq_cov & np.isnan(neq_variance_grid),
            neq_patch.variance,
            neq_variance_grid,
        )
    result: list[PMFPatch] = []
    for eq_patch in eq_patches:
        eq_cov = np.asarray(eq_patch.coverage_mask, dtype=bool) & np.isfinite(eq_patch.pmf)
        synthetic_cov = eq_cov & np.isfinite(neq_variance_grid)
        result.append(PMFPatch(
            name=eq_patch.name,
            kind="EQ_MBAR_with_NEQ_variance",
            root=eq_patch.root,
            grid=grid,
            pmf=eq_patch.pmf,
            variance=np.where(synthetic_cov, neq_variance_grid, np.nan),
            coverage_mask=synthetic_cov,
            source_names=eq_patch.source_names + [p.name for p in neq_patches],
            metadata={
                **eq_patch.metadata,
                "variance_source": "EQ_NEQ_variance",
                "pmf_source": "EQ_MBAR",
            },
        ))
    return result


def eq_network_is_connected(clusters: "list[EQCluster]") -> bool:
    """Return True when all EQ windows have merged into one connected EQ cluster."""
    return len(clusters) == 1


def fallback_after_eq_connectivity_lost(
    *,
    clusters: "list[EQCluster]",
    windows: "list[EnsembleWindow]",
    out_root: "Path",
) -> None:
    """Seed more windows or NEQ bridges when final extension splits the EQ cluster.

    TODO: implement EQ-based or NEQ-based fallback refinement here.
    Currently writes diagnostics and returns; the main workflow marks the run as not converged.
    """
    write_json(out_root / "eq_connectivity_lost.json", {
        "n_clusters_after_extension": len(clusters),
        "cluster_names": [c.name for c in clusters],
        "n_windows": len(windows),
    })


def run_final_eq_extension_refinement(
    *,
    windows: "list[EnsembleWindow]",
    clusters: "list[EQCluster]",
    segment_store: "dict[tuple[str, str], NEQSegment]",
    neq_patch_store: "dict[str, PMFPatch]",
    neq_patch_status: "dict[str, dict[str, Any]]",
    patches_for_global: "list[PMFPatch]",
    skipped_segment_rows: "list[dict[str, Any]]",
    args: "argparse.Namespace",
    ctx: "dict[str, Any]",
    grid: "np.ndarray",
    out_root: "Path",
    bin_path: str,
    budget: "BudgetTracker",
    timing_rows: "list[dict[str, Any]]",
    quality_rows: "list[dict[str, Any]]",
    global_pmf: "np.ndarray",
    global_variance: "np.ndarray",
    fit_details: "dict[str, Any]",
    js_rows: "list[dict[str, Any]]",
    analysis_xmin: float,
    analysis_xmax: float,
) -> "tuple[list[EQCluster], list[PMFPatch], np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], list[PMFPatch], list[dict[str, Any]]]":
    """Run final EQ-extension refinement until target MBAR ddF or budget exhaustion.

    Returns (clusters, patches, global_pmf, global_variance, fit_details, js_rows,
             patches_for_global, extension_summary_rows).
    """
    _EXT_SUMMARY_COLS = [
        "round", "used_steps", "remaining_steps", "n_selected_windows",
        "eq_extension_steps", "round_cost", "max_mbar_ddf", "x_at_max_mbar_ddf",
        "target_mbar_ddf", "stop_reason", "n_clusters_after_extension",
    ]
    ext_rows: list[dict[str, Any]] = []

    def _write_empty_summary(reason: str) -> None:
        ext_rows.append({
            "round": 0, "used_steps": int(budget.used_steps),
            "remaining_steps": int(budget.total_budget_steps - budget.used_steps),
            "n_selected_windows": 0, "eq_extension_steps": 0,
            "round_cost": 0, "max_mbar_ddf": float("nan"),
            "x_at_max_mbar_ddf": float("nan"),
            "target_mbar_ddf": float(args.target_mbar_ddf),
            "stop_reason": reason,
            "n_clusters_after_extension": len(clusters),
        })
        write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)

    if getattr(args, "final_refinement_mode", "none") != "eq-extend":
        _write_empty_summary("mode_disabled")
        return clusters, patches_for_global, global_pmf, global_variance, fit_details, js_rows, patches_for_global, ext_rows

    if not eq_network_is_connected(clusters):
        _write_empty_summary("eq_network_not_connected")
        return clusters, patches_for_global, global_pmf, global_variance, fit_details, js_rows, patches_for_global, ext_rows

    eq_ext_steps = int(getattr(args, "eq_extension_steps", None) or args.n_eq_steps)
    target_ddf = float(getattr(args, "target_mbar_ddf", 1.0e-3))
    ext_root = out_root / "final_eq_extension"
    ext_root.mkdir(parents=True, exist_ok=True)
    base_seed = int(args.seed)
    round_index = 0

    analysis_mask = (
        (grid >= float(analysis_xmin))
        & (grid <= float(analysis_xmax))
        & np.isfinite(global_variance)
    )

    def _compute_max_mbar_ddf(gvar: np.ndarray) -> tuple[float, float]:
        valid = analysis_mask & np.isfinite(gvar) & (gvar >= 0.0)
        if not valid.any():
            return float("nan"), float("nan")
        ddf_arr = np.sqrt(gvar[valid])
        idx = int(np.argmax(ddf_arr))
        return float(ddf_arr[idx]), float(grid[valid][idx])

    while True:
        max_ddf, x_at_max = _compute_max_mbar_ddf(global_variance)
        selected_windows = list(clusters[0].windows)
        round_cost = len(selected_windows) * eq_ext_steps
        remaining = int(budget.total_budget_steps) - int(budget.used_steps)

        summary_row: dict[str, Any] = {
            "round": round_index,
            "used_steps": int(budget.used_steps),
            "remaining_steps": remaining,
            "n_selected_windows": len(selected_windows),
            "eq_extension_steps": eq_ext_steps,
            "round_cost": round_cost,
            "max_mbar_ddf": max_ddf,
            "x_at_max_mbar_ddf": x_at_max,
            "target_mbar_ddf": target_ddf,
            "stop_reason": "",
            "n_clusters_after_extension": 1,
        }

        if np.isfinite(max_ddf) and max_ddf < target_ddf:
            summary_row["stop_reason"] = "target_mbar_ddf_reached"
            ext_rows.append(summary_row)
            break

        if not budget.can_spend(round_cost):
            summary_row["stop_reason"] = "budget_exhausted"
            ext_rows.append(summary_row)
            break

        round_root = ext_root / f"round_{round_index:03d}"
        round_root.mkdir(parents=True, exist_ok=True)

        for wi, window in enumerate(selected_windows):
            ext_win_root = round_root / "windows" / window.name
            ext_win_root.mkdir(parents=True, exist_ok=True)
            ext_seed = base_seed + 200000 + round_index * 10000 + wi
            nout = max(1, int(math.ceil(float(eq_ext_steps) / max(int(args.eq_save_every), 1))))
            run_eq_window_raw(
                bin_path=bin_path,
                ctx=ctx,
                center_x=float(window.center_x),
                k=float(window.k),
                steps=eq_ext_steps,
                nout=nout,
                seed=ext_seed,
                out_dir=ext_win_root,
            )
            ext_eq_file = ext_win_root / "eq_window.csv"
            ext_eq_rows = read_csv_rows(ext_eq_file)
            ext_tail_rows = tail_rows_from_eq_rows(ext_eq_rows, float(args.tail_fraction))
            ext_tail_file = ext_win_root / "eq_tail.csv"
            write_csv(ext_tail_file, ordered_fieldnames(ext_eq_rows or window.eq_rows), ext_tail_rows)
            combined_tail_rows = list(window.tail_rows) + list(ext_tail_rows)
            combined_tail_file = round_root / "windows" / window.name / "eq_tail_combined.csv"
            write_csv(combined_tail_file, ordered_fieldnames(combined_tail_rows), combined_tail_rows)
            window.tail_rows = combined_tail_rows
            window.tail_file = combined_tail_file
            tail_x_arr = np.asarray(
                [float(r["x"]) for r in combined_tail_rows if r.get("x", "") != ""],
                dtype=float,
            )
            tail_x_finite = tail_x_arr[np.isfinite(tail_x_arr)]
            if tail_x_finite.size >= 2:
                window.mean_x = float(np.mean(tail_x_finite))
                window.std_x = float(np.std(tail_x_finite, ddof=1))
                window.x_most = float(mode_x_from_samples(tail_x_finite, grid))

        budget.spend(round_cost, f"final_eq_extension_round_{round_index:03d}", "EQ_EXTENSION", "final_eq_extension")

        stage_label = f"final_eq_extension_round_{round_index:03d}"
        with timed_operation(timing_rows, stage=stage_label, operation="rebuild_clusters_and_pmf", item="all"):
            new_clusters, new_js_rows = build_eq_clusters(windows, grid, float(args.js_threshold), ctx=ctx)
            patch_root_ext = round_root / "patches"
            new_patches: list[PMFPatch] = []
            for ci, cl in enumerate(new_clusters):
                new_patches.append(
                    build_eq_cluster_patch(
                        cl, grid, ctx, int(args.n_bootstrap_eq), patch_root_ext,
                        rng_seed=base_seed + 300000 + round_index * 1000 + ci,
                    )
                )
            new_global_pmf, new_global_variance, new_fit_details = fit_global_pmf_from_patches(
                new_patches, grid, float(args.variance_floor), reference_x=None,
            )

        if not eq_network_is_connected(new_clusters):
            summary_row["stop_reason"] = "eq_connectivity_lost"
            summary_row["n_clusters_after_extension"] = len(new_clusters)
            ext_rows.append(summary_row)
            write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)
            return clusters, patches_for_global, global_pmf, global_variance, fit_details, js_rows, patches_for_global, ext_rows

        clusters = new_clusters
        js_rows = new_js_rows
        patches_for_global = new_patches
        global_pmf = new_global_pmf
        global_variance = new_global_variance
        fit_details = new_fit_details

        all_segs_sorted = sorted(segment_store.values(), key=lambda s: s.name)
        write_state_tables(
            round_root, out_root, windows, clusters, all_segs_sorted,
            patches_for_global, fit_details, js_rows, grid, global_pmf, global_variance, ctx,
        )

        quality_rows.append(compute_pmf_quality_metrics(
            grid=grid, global_pmf=global_pmf, global_variance=global_variance,
            ctx=ctx, used_steps=int(budget.used_steps), stage=stage_label,
            analysis_xmin=float(analysis_xmin), analysis_xmax=float(analysis_xmax),
        ))

        analysis_mask = (
            (grid >= float(analysis_xmin))
            & (grid <= float(analysis_xmax))
            & np.isfinite(global_variance)
        )
        summary_row["stop_reason"] = "round_complete"
        ext_rows.append(summary_row)
        write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)
        round_index += 1

    write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)
    return clusters, patches_for_global, global_pmf, global_variance, fit_details, js_rows, patches_for_global, ext_rows


def reconstruct_chain(
    *,
    windows: list[EnsembleWindow],
    segment_store: dict[tuple[str, str], NEQSegment],
    neq_patch_store: dict[str, PMFPatch],
    neq_patch_status: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    ctx: dict[str, Any],
    grid: np.ndarray,
    out_root: Path,
    bin_path: str,
    base_seed: int,
    stage_index: int,
    budget: BudgetTracker,
    stage_label: str,
    skipped_segment_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[EQCluster], list[NEQSegment], list[PMFPatch], np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], list[PMFPatch]]:
    _timing: list[dict[str, Any]] = timing_rows if timing_rows is not None else []
    clusters, js_rows = build_eq_clusters(windows, grid, float(args.js_threshold), ctx=ctx)
    patch_root = out_root / "patches"
    segments: list[NEQSegment] = []
    patches: list[PMFPatch] = []
    for cluster_idx, cluster in enumerate(clusters):
        with timed_operation(
            _timing,
            stage=stage_label,
            operation="build_eq_cluster_patch",
            item=cluster.name,
            metadata={"n_windows": len(cluster.windows), "stage_index": stage_index},
        ):
            patches.append(
                build_eq_cluster_patch(
                    cluster,
                    grid,
                    ctx,
                    int(args.n_bootstrap_eq),
                    patch_root,
                    rng_seed=int(base_seed + 1000 * stage_index + cluster_idx),
                )
            )
    for cluster_idx in range(len(clusters) - 1):
        left_cluster = clusters[cluster_idx]
        right_cluster = clusters[cluster_idx + 1]
        left_boundary, right_boundary, _boundary_meta = choose_connected_boundary_pair(
            left_cluster, right_cluster, segment_store
        )
        key = (left_boundary.name, right_boundary.name)
        segment = segment_store.get(key)
        if segment is None:
            segment_name = f"SEG_{left_boundary.name}__{right_boundary.name}"
            segment_root = out_root / "segments" / segment_name
            neq_budget = choose_affordable_neq_count(
                requested_n_neq=int(args.n_neq_traj),
                t_neq=int(args.t_neq),
                budget=budget,
                allow_partial=bool(args.allow_partial_neq_budget),
                min_neq_traj=int(args.min_neq_traj),
            )
            if not bool(neq_budget["can_run"]):
                skipped_segment_rows.append(
                    {
                        "segment": segment_name,
                        "left_boundary": left_boundary.name,
                        "right_boundary": right_boundary.name,
                        "reason": str(neq_budget["reason"]),
                        "n_neq_traj_requested": int(neq_budget["n_neq_traj_requested"]),
                        "n_neq_traj_affordable": int(neq_budget["n_neq_traj_actual"]),
                        "min_neq_traj": int(neq_budget["min_neq_traj"]),
                        "cost_requested": int(neq_budget["cost_requested"]),
                        "remaining_budget_before_segment": int(neq_budget["remaining_budget_before_segment"]),
                        "used_steps": int(budget.used_steps),
                        "budget_steps": int(budget.total_budget_steps),
                    }
                )
                continue
            budget.spend(
                int(neq_budget["cost_actual"]),
                segment_name,
                "NEQ_PARTIAL" if bool(neq_budget["neq_budget_limited"]) else "NEQ",
                stage_label,
            )
            with timed_operation(
                _timing,
                stage=stage_label,
                operation="run_neq_protocol",
                item=segment_name,
                metadata={
                    "n_neq_traj": int(neq_budget["n_neq_traj_actual"]),
                    "t_neq": int(args.t_neq),
                    "stage_index": stage_index,
                },
            ):
                segment = run_neq_protocol(
                    name=segment_name,
                    left=left_cluster,
                    right=right_cluster,
                    boundary_left=left_boundary,
                    boundary_right=right_boundary,
                    bin_path=bin_path,
                    ctx=ctx,
                    t_neq=int(args.t_neq),
                    n_neq_traj=int(neq_budget["n_neq_traj_actual"]),
                    seed=int(base_seed + 400000 + 1000 * stage_index + cluster_idx),
                    root=segment_root,
                    k_min=float(args.k_min),
                    k_max=float(args.k_max),
                    out_root=out_root,
                    neq_protocol_mode=str(args.neq_protocol_mode),
                    n_neq_traj_requested=int(neq_budget["n_neq_traj_requested"]),
                    neq_budget_limited=bool(neq_budget["neq_budget_limited"]),
                    neq_cost_requested=int(neq_budget["cost_requested"]),
                    neq_cost_actual=int(neq_budget["cost_actual"]),
                    remaining_budget_before_segment=int(neq_budget["remaining_budget_before_segment"]),
                )
            segment_store[key] = segment
        segments.append(segment)
        connectivity = ensure_segment_connectivity(
            segment,
            grid,
            float(args.neq_connectivity_threshold),
            out_root,
            mts_patch_built=False,
            reason="connectivity_diagnostic_only",
        )
        if segment.name in neq_patch_store:
            # Reuse previously built patch — skip expensive bootstrapping
            _neq_patch = neq_patch_store[segment.name]
            patches.append(_neq_patch)
            ensure_segment_connectivity(
                segment,
                grid,
                float(args.neq_connectivity_threshold),
                out_root,
                mts_patch_built=True,
                reason="reused_valid_neq_mts_patch_from_store",
            )
        else:
            try:
                _neq_patch = None
                with timed_operation(
                    _timing,
                    stage=stage_label,
                    operation="build_neq_mts_patch",
                    item=segment.name,
                    metadata={"stage_index": stage_index},
                ):
                    _neq_patch = build_neq_mts_patch(
                        segment,
                        grid,
                        ctx,
                        int(args.n_bootstrap_neq),
                        patch_root,
                        rng_seed=int(base_seed + 600000 + 1000 * stage_index + cluster_idx),
                        disable_fixed_cft_bootstrap=bool(args.disable_fixed_cft_bootstrap),
                    )
                neq_patch_store[segment.name] = _neq_patch
                neq_patch_status[segment.name] = {
                    "segment": segment.name,
                    "left_boundary": segment.left_boundary.name,
                    "right_boundary": segment.right_boundary.name,
                    "mts_patch_built": 1,
                    "reused": 0,
                    "reason": "valid_neq_mts_patch_persisted",
                }
                patches.append(_neq_patch)
                ensure_segment_connectivity(
                    segment,
                    grid,
                    float(args.neq_connectivity_threshold),
                    out_root,
                    mts_patch_built=True,
                    reason=(
                        "built_neq_mts_patch_eop_crossed"
                        if bool(connectivity.get("eop_crossed", False))
                        else "built_neq_mts_patch_without_eop_crossing"
                    ),
                )
                if segment.neq_patch_decision:
                    segment.neq_patch_decision["reason"] = (
                        "built_neq_mts_patch_eop_crossed"
                        if bool(connectivity.get("eop_crossed", False))
                        else "built_neq_mts_patch_without_eop_crossing"
                    )
            except Exception as exc:
                segment.mts_patch_built = False
                if not segment.neq_patch_decision:
                    segment.neq_patch_decision = {
                        "segment": segment.name,
                        "left_boundary": segment.left_boundary.name,
                        "right_boundary": segment.right_boundary.name,
                        "eop_crossed": connectivity.get("eop_crossed", ""),
                        "eop_jsd_raw": connectivity.get("eop_jsd_raw", ""),
                        "eop_jsd_norm": connectivity.get("eop_jsd_norm", ""),
                        "eop_common_bins": connectivity.get("eop_common_bins", ""),
                        "eop_interval_overlap": connectivity.get("eop_interval_overlap", ""),
                        "mts_patch_built": 0,
                        "included_in_global_pmf": 0,
                        "cft_solved_once": int(bool(segment.cft_summary.get("cft_solved_once", False))),
                        "cft_delta_f": segment.cft_summary.get("cft_delta_f", ""),
                        "cft_delta_f_unc": segment.cft_summary.get("cft_delta_f_unc", ""),
                        "bootstrap_recomputed_cft": int(bool(segment.cft_summary.get("bootstrap_recomputed_cft", False))),
                        "reason": f"failed_to_build_neq_mts_patch: {exc}",
                    }
                ensure_segment_connectivity(
                    segment,
                    grid,
                    float(args.neq_connectivity_threshold),
                    out_root,
                    mts_patch_built=False,
                    reason=f"failed_to_build_neq_mts_patch: {exc}",
                )
    pmf_method = str(getattr(args, "pmf_method", "hybrid"))
    eq_patches_current = [p for p in patches if p.kind == "EQ_MBAR"]
    # Use ALL stored NEQ patches (not just ones built this round)
    neq_patches_from_store = [p for p in neq_patch_store.values() if int(np.count_nonzero(p.coverage_mask)) > 0]
    # When all EQ windows form one connected cluster, use EQ-MBAR only.
    # NEQ/MTS patches are excluded from the global PMF fit and retained only for diagnostics.
    if eq_network_is_connected(clusters):
        patches_for_global = eq_patches_current
        patch_selection_rule = "connected_EQ_MBAR_only"
        variance_source = "EQ_MBAR_bootstrap"
    else:
        if pmf_method == "neq":
            if not neq_patches_from_store:
                raise RuntimeError(
                    "pmf_method=neq but no NEQ_MTS patches are available. "
                    "Check that NEQ simulations ran and produced valid patches."
                )
            patches_for_global = neq_patches_from_store
        elif pmf_method == "hybrid":
            patches_for_global = eq_patches_current + neq_patches_from_store
        elif pmf_method == "eq":
            patches_for_global = build_eq_pmf_with_neq_variance(
                eq_patches=eq_patches_current,
                neq_patches=neq_patches_from_store,
                grid=grid,
            )
        else:
            raise ValueError(f"Unknown pmf_method: {pmf_method!r}")
        patch_selection_rule = {
            "neq": "only_NEQ_MTS",
            "hybrid": "EQ_MBAR_plus_NEQ_MTS",
            "eq": "EQ_MBAR_pmf_with_EQ_NEQ_variance",
        }[pmf_method]
        variance_source = {
            "neq": "NEQ_MTS_bootstrap",
            "hybrid": "hybrid_patch_variance",
            "eq": "EQ_NEQ_variance",
        }[pmf_method]
    # Classify all stored NEQ patches against current cluster graph
    _current_neighbor_names = {s.name for s in segments}
    _window_to_cluster = build_window_to_cluster_map(clusters)
    _patches_for_global_names = {p.name for p in patches_for_global}
    for _seg_name, _stored_patch in neq_patch_store.items():
        _st = neq_patch_status.setdefault(_seg_name, {})
        _left_b = _st.get("left_boundary", "")
        _right_b = _st.get("right_boundary", "")
        _left_c = _window_to_cluster.get(_left_b, "")
        _right_c = _window_to_cluster.get(_right_b, "")
        _is_current = _seg_name in _current_neighbor_names
        _is_internal = bool(_left_c and _right_c and _left_c == _right_c)
        _is_long_range = bool(_left_c and _right_c and _left_c != _right_c and not _is_current)
        _st.update({
            "is_current_neighbor_edge": int(_is_current),
            "is_internal_to_current_eq_cluster": int(_is_internal),
            "is_long_range_or_obsolete_edge": int(_is_long_range),
            "left_current_cluster": _left_c,
            "right_current_cluster": _right_c,
            "included_in_global_fit": int(_stored_patch.name in _patches_for_global_names),
            "n_covered_bins": int(np.count_nonzero(_stored_patch.coverage_mask)),
        })

    with timed_operation(
        _timing,
        stage=stage_label,
        operation="fit_global_pmf_from_patches",
        item="global",
        metadata={"n_patches": len(patches_for_global), "stage_index": stage_index},
    ):
        global_pmf, global_variance, fit_details = fit_global_pmf_from_patches(
            patches_for_global,
            grid,
            float(args.variance_floor),
            reference_x=None,
        )
    fit_details["pmf_method"] = pmf_method
    fit_details["patch_selection_rule"] = patch_selection_rule
    fit_details["variance_source"] = variance_source
    fit_details["eq_network_connected"] = bool(eq_network_is_connected(clusters))
    fit_details["final_estimator"] = (
        "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
    )
    write_patch_offset_aligned_outputs(patches_for_global, fit_details)
    write_all_neq_patches_csv(out_root, neq_patch_store, neq_patch_status, out_root)
    write_patches_used_for_global_fit_csv(out_root, patches_for_global, fit_details, neq_patch_store, neq_patch_status, out_root)
    return clusters, segments, patches, global_pmf, global_variance, fit_details, js_rows, patches_for_global


def stage_cost_eq(steps: int) -> int:
    return int(steps)


def stage_cost_neq(n_neq_traj: int, t_neq: int) -> int:
    return int(2 * n_neq_traj * t_neq)


def choose_rescue_target(
    grid: np.ndarray,
    global_variance: np.ndarray,
    analysis_xmin: float | None = None,
    analysis_xmax: float | None = None,
) -> tuple[float, float]:
    finite = np.isfinite(global_variance)
    if analysis_xmin is not None:
        finite &= np.asarray(grid, dtype=float) >= float(analysis_xmin)
    if analysis_xmax is not None:
        finite &= np.asarray(grid, dtype=float) <= float(analysis_xmax)
    if not np.any(finite):
        return float("nan"), float("nan")
    finite_indices = np.where(finite)[0]
    best_local_idx = int(np.nanargmax(global_variance[finite]))
    best_idx = int(finite_indices[best_local_idx])
    return float(grid[best_idx]), float(global_variance[best_idx])


def find_uncovered_intervals(
    grid: np.ndarray,
    uncovered_mask: np.ndarray,
) -> list[dict[str, Any]]:
    indices = np.where(np.asarray(uncovered_mask, dtype=bool))[0]
    if indices.size <= 0:
        return []

    def make_interval(start_idx: int, end_idx: int) -> dict[str, Any]:
        x_start = float(grid[start_idx])
        x_end = float(grid[end_idx])
        midpoint = 0.5 * (x_start + x_end)
        return {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "x_start": x_start,
            "x_end": x_end,
            "width": float(max(0.0, x_end - x_start)),
            "n_bins": int(end_idx - start_idx + 1),
            "midpoint": float(midpoint),
        }

    intervals: list[dict[str, Any]] = []
    start_idx = int(indices[0])
    prev_idx = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev_idx + 1:
            prev_idx = idx
            continue
        intervals.append(make_interval(start_idx, prev_idx))
        start_idx = idx
        prev_idx = idx
    intervals.append(make_interval(start_idx, prev_idx))
    return intervals


def choose_uncovered_rescue_target(
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: np.ndarray,
    analysis_xmin: float,
    analysis_xmax: float,
) -> dict[str, Any] | None:
    del global_variance
    grid_arr = np.asarray(grid, dtype=float)
    half_dx = 0.5 * abs(float(grid_arr[1] - grid_arr[0])) if len(grid_arr) > 1 else 0.05
    bin_width = 2.0 * half_dx
    analysis_mask = (
        np.isfinite(grid_arr)
        & (grid_arr >= float(analysis_xmin) - half_dx)
        & (grid_arr <= float(analysis_xmax) + half_dx)
    )
    covered_mask = analysis_mask & np.isfinite(global_pmf)
    uncovered_mask = analysis_mask & ~covered_mask
    intervals = find_uncovered_intervals(grid_arr, uncovered_mask)
    if not intervals:
        return None

    min_uncovered_bins = 2
    min_uncovered_width = 0.75 * bin_width

    max_width = max(float(row["width"]) for row in intervals)
    width_candidates = [row for row in intervals if float(row["width"]) == float(max_width)]
    max_n_bins = max(int(row["n_bins"]) for row in width_candidates)
    bin_candidates = [row for row in width_candidates if int(row["n_bins"]) == int(max_n_bins)]
    analysis_center = 0.5 * (float(analysis_xmin) + float(analysis_xmax))
    best = min(bin_candidates, key=lambda row: abs(float(row["midpoint"]) - analysis_center))

    if int(best["n_bins"]) < min_uncovered_bins and float(best["width"]) < min_uncovered_width:
        return {
            "target_x": float("nan"),
            "target_variance": float("nan"),
            "target_priority": "uncovered_interval_ignored",
            "target_reason": "below_min_uncovered_rescue_size",
            "uncovered_start_x": float("nan"),
            "uncovered_end_x": float("nan"),
            "uncovered_width": float("nan"),
            "uncovered_n_bins": 0,
            "ignored_uncovered_start_x": float(best["x_start"]),
            "ignored_uncovered_end_x": float(best["x_end"]),
            "ignored_uncovered_n_bins": int(best["n_bins"]),
            "ignored_uncovered_width": float(best["width"]),
            "ignored_uncovered_reason": "below_min_uncovered_rescue_size",
        }

    target_x = min(max(float(best["midpoint"]), float(analysis_xmin)), float(analysis_xmax))
    return {
        "target_x": target_x,
        "target_variance": float("nan"),
        "target_priority": "uncovered_interval",
        "target_reason": "widest_uncovered_interval",
        "uncovered_start_x": float(best["x_start"]),
        "uncovered_end_x": float(best["x_end"]),
        "uncovered_width": float(best["width"]),
        "uncovered_n_bins": int(best["n_bins"]),
    }


def choose_failed_or_skipped_gap_target(
    *,
    analysis_xmin: float,
    analysis_xmax: float,
    global_pmf: np.ndarray,
    grid: np.ndarray,
    skipped_segment_rows: list[dict[str, Any]] | None = None,
    mts_failed_segments: list[NEQSegment] | None = None,
) -> dict[str, Any] | None:
    del skipped_segment_rows
    if not mts_failed_segments:
        return None

    analysis_lo = float(analysis_xmin)
    analysis_hi = float(analysis_xmax)
    candidates: list[dict[str, Any]] = []
    for segment in mts_failed_segments:
        left_x = float(segment.left_boundary.mean_x)
        right_x = float(segment.right_boundary.mean_x)
        lo = min(left_x, right_x)
        hi = max(left_x, right_x)
        midpoint = 0.5 * (lo + hi)
        if midpoint < analysis_lo or midpoint > analysis_hi:
            continue
        interval_mask = (
            np.isfinite(grid)
            & (grid >= lo - 1.0e-9)
            & (grid <= hi + 1.0e-9)
        )
        if not np.any(interval_mask):
            continue
        interval_uncovered = interval_mask & ~np.isfinite(global_pmf)
        if not np.any(interval_uncovered):
            continue
        candidates.append(
            {
                "target_x": float(midpoint),
                "target_variance": float("nan"),
                "target_priority": "failed_or_skipped_segment",
                "target_reason": "skipped_or_mts_failed_gap",
                "uncovered_start_x": float(lo),
                "uncovered_end_x": float(hi),
                "uncovered_width": float(max(0.0, hi - lo)),
                "uncovered_n_bins": int(np.count_nonzero(interval_uncovered)),
                "gap_coordinate": "mean_x",
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (float(row["uncovered_width"]), int(row["uncovered_n_bins"])),
    )


def choose_rescue_target_priority(
    *,
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_variance: np.ndarray,
    analysis_xmin: float,
    analysis_xmax: float,
    skipped_segment_rows: list[dict[str, Any]] | None = None,
    mts_failed_segments: list[NEQSegment] | None = None,
) -> dict[str, Any] | None:
    uncovered_target = choose_uncovered_rescue_target(
        grid,
        global_pmf,
        global_variance,
        analysis_xmin,
        analysis_xmax,
    )
    if uncovered_target is not None and str(uncovered_target.get("target_priority", "")) != "uncovered_interval_ignored":
        return uncovered_target
    _ignored_uncovered_meta: dict[str, Any] = (
        {k: v for k, v in uncovered_target.items() if k.startswith("ignored_")}
        if uncovered_target is not None
        else {}
    )

    failed_or_skipped_target = choose_failed_or_skipped_gap_target(
        analysis_xmin=float(analysis_xmin),
        analysis_xmax=float(analysis_xmax),
        global_pmf=global_pmf,
        grid=grid,
        skipped_segment_rows=skipped_segment_rows,
        mts_failed_segments=mts_failed_segments,
    )
    if failed_or_skipped_target is not None:
        return failed_or_skipped_target

    target_x, target_variance = choose_rescue_target(
        grid,
        global_variance,
        analysis_xmin=float(analysis_xmin),
        analysis_xmax=float(analysis_xmax),
    )
    if not math.isfinite(target_x) or not math.isfinite(target_variance):
        return None
    return {
        "target_x": float(target_x),
        "target_variance": float(target_variance),
        "target_priority": "finite_variance",
        "target_reason": "max_finite_global_variance",
        "uncovered_start_x": float("nan"),
        "uncovered_end_x": float("nan"),
        "uncovered_width": float("nan"),
        "uncovered_n_bins": int(0),
        **_ignored_uncovered_meta,
    }


def finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def load_child_design_records(generations_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    design_files = sorted(generations_root.glob("g*/left/child_design.json")) + sorted(
        generations_root.glob("g*/right/child_design.json")
    )
    for design_file in design_files:
        try:
            raw = load_json(design_file)
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        target_x = finite_float_or_none(raw.get("target_x"))
        center_x = finite_float_or_none(raw.get("center_x"))
        k = finite_float_or_none(raw.get("k"))
        if target_x is None or center_x is None or k is None:
            continue
        records.append(
            {
                "design_file": str(design_file),
                "name": str(raw.get("name", "")),
                "side": str(raw.get("side", "")),
                "target_x": target_x,
                "center_x": center_x,
                "k": k,
                "raw_k": raw.get("raw_k", ""),
                "k_rule": raw.get("k_rule", ""),
                "target_source": raw.get("target_source", ""),
                "barrier_crossing_diagnostic": raw.get("barrier_crossing_diagnostic", raw.get("barrier_crossing", "")),
            }
        )
    return records


def match_child_design_for_rescue_target(
    child_designs: list[dict[str, Any]],
    x_rescue_target: float,
) -> dict[str, Any] | None:
    if not child_designs:
        return None
    return min(
        child_designs,
        key=lambda record: abs(float(record["target_x"]) - float(x_rescue_target)),
    )


def count_previous_rescue_retries(
    rescue_rows: list[dict[str, Any]],
    *,
    x_rescue_target: float,
    matched_child_name: str | None,
    grid_dx: float,
) -> int:
    count = 0
    for row in rescue_rows:
        prev_name = str(row.get("matched_child_name", ""))
        same_child = (
            matched_child_name is not None
            and matched_child_name != ""
            and prev_name == str(matched_child_name)
        )
        prev_x = finite_float_or_none(row.get("x_rescue_target"))
        same_target = prev_x is not None and abs(prev_x - float(x_rescue_target)) <= float(grid_dx)
        if same_child or same_target:
            prev_contains = row.get("rescue_tail_contains_target_bin")
            prev_fraction = finite_float_or_none(row.get("target_bin_tail_fraction"))
            failed_sampling = (
                prev_contains is False
                or prev_contains == 0
                or (prev_fraction is not None and prev_fraction < 0.05)
            )
            if failed_sampling:
                count += 1
    return count


def clamp_to_bounds(
    value: float,
    lower: float,
    upper: float,
) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def design_rescue_window(
    *,
    target_info: dict[str, Any],
    generations_root: Path,
    grid: np.ndarray,
    args: argparse.Namespace,
    analysis_xmin: float,
    analysis_xmax: float,
    rescue_rows: list[dict[str, Any]] | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_rescue_target = float(target_info["target_x"])
    grid_arr = np.asarray(grid, dtype=float)
    grid_dx = abs(float(grid_arr[1] - grid_arr[0])) if len(grid_arr) > 1 else float(args.bin_width)
    bin_width = grid_dx

    target_bin_index = int(np.argmin(np.abs(grid_arr - x_rescue_target)))
    target_bin_x = float(grid_arr[target_bin_index])
    rescue_center_x_raw = target_bin_x
    rescue_center_x = clamp_to_bounds(rescue_center_x_raw, analysis_xmin, analysis_xmax)
    rescue_center_clamped = abs(rescue_center_x - rescue_center_x_raw) > 0.5 * grid_dx
    rescue_center_rule = "target_bin_x_clamped_to_analysis_bounds"

    child_designs = load_child_design_records(generations_root)
    matched = match_child_design_for_rescue_target(child_designs, x_rescue_target)

    n_retry = count_previous_rescue_retries(
        rescue_rows or [],
        x_rescue_target=x_rescue_target,
        matched_child_name=str(matched["name"]) if matched else None,
        grid_dx=grid_dx,
    )

    kT = float(ctx["thermal_kT"]) if ctx is not None else 1.0
    sigma_target = max(1.5 * bin_width, 0.20)
    k_from_sigma = kT / (sigma_target ** 2)
    rescue_k_base = max(float(args.k_rescue), k_from_sigma)
    rescue_scale = float(args.s_rescue) ** float(n_retry)
    rescue_k_unclipped = rescue_k_base * rescue_scale
    rescue_k = min(max(rescue_k_unclipped, float(args.k_min)), float(args.k_max))
    rescue_k_saturated = rescue_k >= float(args.k_max) - 1.0e-12
    rescue_k_rule = "target_bin_sigma_rule_with_retry_scaling"

    return {
        "x_rescue_target": float(x_rescue_target),
        "target_bin_x": float(target_bin_x),
        "target_bin_index": int(target_bin_index),
        "rescue_center_x_raw": float(rescue_center_x_raw),
        "rescue_center_x": float(rescue_center_x),
        "rescue_center_clamped_to_bounds": bool(rescue_center_clamped),
        "rescue_center_rule": rescue_center_rule,
        "rescue_center_f_raw": float("nan"),
        "rescue_center_f": float("nan"),
        "sigma_target": float(sigma_target),
        "k_from_sigma": float(k_from_sigma),
        "rescue_k_base": float(rescue_k_base),
        "rescue_k": float(rescue_k),
        "rescue_k_unclipped": float(rescue_k_unclipped),
        "rescue_k_saturated": bool(rescue_k_saturated),
        "rescue_k_rule": rescue_k_rule,
        "s_rescue": float(args.s_rescue),
        "rescue_retry_count": int(n_retry),
        "rescue_scale": float(rescue_scale),
        "matched_child_design": str(matched["design_file"]) if matched else "",
        "matched_child_name": str(matched["name"]) if matched else "",
        "matched_child_side": str(matched["side"]) if matched else "",
        "matched_target_x": float(matched["target_x"]) if matched else "",
        "matched_target_distance": abs(float(matched["target_x"]) - x_rescue_target) if matched else "",
        "matched_child_center_x": float(matched["center_x"]) if matched else "",
        "matched_child_k": float(matched["k"]) if matched else "",
        "matched_child_raw_k": matched.get("raw_k", "") if matched else "",
        "matched_child_k_rule": matched.get("k_rule", "") if matched else "",
        "matched_child_target_source": matched.get("target_source", "") if matched else "",
        "matched_child_used_for_center": 0,
    }


def design_gt_rescue_window(
    *,
    target_info: dict[str, Any],
    clusters: list[EQCluster],
    segment_store: dict[tuple[str, str], "NEQSegment"],
    grid: np.ndarray,
    args: argparse.Namespace,
    analysis_xmin: float,
    analysis_xmax: float,
    rescue_rows: list[dict[str, Any]] | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_target = float(target_info["target_x"])
    grid_arr = np.asarray(grid, dtype=float)
    grid_dx = abs(float(grid_arr[1] - grid_arr[0])) if len(grid_arr) > 1 else float(args.bin_width)
    target_bin_index = int(np.argmin(np.abs(grid_arr - x_target)))
    target_bin_x = float(grid_arr[target_bin_index])

    gt_left_cluster: EQCluster | None = None
    gt_right_cluster: EQCluster | None = None

    gap_pair = gap_clusters_for_target(x_target, clusters)
    if gap_pair is not None:
        gt_left_cluster, gt_right_cluster = gap_pair
    else:
        covering = cluster_covering_target(x_target, clusters)
        if covering is not None:
            cidx = next((i for i, c in enumerate(clusters) if c.name == covering.name), None)
            if cidx is not None and cidx > 0:
                gt_left_cluster = clusters[cidx - 1]
                gt_right_cluster = covering
            elif cidx is not None and cidx < len(clusters) - 1:
                gt_left_cluster = covering
                gt_right_cluster = clusters[cidx + 1]

    if gt_left_cluster is None or gt_right_cluster is None:
        rescue_center_x_raw = target_bin_x
        rescue_center_x = clamp_to_bounds(rescue_center_x_raw, analysis_xmin, analysis_xmax)
        kT = float(ctx["thermal_kT"]) if ctx is not None else 1.0
        sigma_target = max(1.5 * grid_dx, 0.20)
        k_from_sigma = kT / (sigma_target ** 2)
        rescue_k_base = max(float(args.k_rescue), k_from_sigma)
        rescue_k = min(max(rescue_k_base, float(args.k_min)), float(args.k_max))
        return {
            "x_rescue_target": float(x_target),
            "target_bin_x": float(target_bin_x),
            "target_bin_index": int(target_bin_index),
            "rescue_center_x_raw": float(rescue_center_x_raw),
            "rescue_center_x": float(rescue_center_x),
            "rescue_center_clamped_to_bounds": bool(abs(rescue_center_x - rescue_center_x_raw) > 0.5 * grid_dx),
            "rescue_center_rule": "target_bin_fallback_no_bracketing_clusters",
            "rescue_k": float(rescue_k),
            "rescue_k_rule": "target_bin_sigma_fallback_no_bracketing_clusters",
            "rescue_k_retry_rule": "disabled_for_GT",
            "rescue_k_unclipped": float(rescue_k_base),
            "rescue_k_saturated": bool(rescue_k >= float(args.k_max) - 1.0e-12),
            "rescue_k_base": float(rescue_k_base),
            "sigma_target": float(sigma_target),
            "k_from_sigma": float(k_from_sigma),
            "rescue_retry_count": 0,
            "rescue_scale": 1.0,
            "matched_child_name": "",
            "matched_child_side": "",
            "matched_child_center_x": float("nan"),
            "matched_child_target_x": float("nan"),
            "matched_target_distance": float("nan"),
            "matched_child_k": float("nan"),
            "matched_child_raw_k": float("nan"),
            "matched_child_k_rule": "",
            "matched_child_target_source": "",
            "matched_child_used_for_center": 0,
            "gt_left_cluster": "",
            "gt_right_cluster": "",
            "gt_left_boundary": "",
            "gt_right_boundary": "",
            "gt_left_center_x": float("nan"),
            "gt_right_center_x": float("nan"),
            "gt_left_mean_x": float("nan"),
            "gt_right_mean_x": float("nan"),
            "gt_left_std_x": float("nan"),
            "gt_right_std_x": float("nan"),
            "gt_left_x_most": float("nan"),
            "gt_right_x_most": float("nan"),
            "gt_m_L": float("nan"),
            "gt_m_R": float("nan"),
            "gt_sigma_L": float("nan"),
            "gt_sigma_R": float("nan"),
            "gt_s_eff": float("nan"),
            "gt_x_raw": float("nan"),
            "gt_k_raw": float("nan"),
            "gt_x0_L": float("nan"),
            "gt_k0_L": float("nan"),
            "gt_x0_R": float("nan"),
            "gt_k0_R": float("nan"),
            "gt_used_midpoint_fallback": False,
            "gt_fallback_reason": "no_bracketing_cluster_pair",
            "gt_boundary_pair_reason": "",
            "gt_anchor_coordinate": "mean_x",
        }

    left_boundary, right_boundary, boundary_meta = choose_connected_boundary_pair(
        gt_left_cluster, gt_right_cluster, segment_store
    )
    EQ_L = eq_gt_tuple(left_boundary)
    EQ_R = eq_gt_tuple(right_boundary)
    x0_L, k0_L, x0_R, k0_R, harmonic_meta = get_k0_x0_harmonic_fromEQ(EQ_L, EQ_R)
    rescue_center_x_gt, rescue_k_gt, gt_ms_meta = get_xs_ks_from_ms(
        x0_L,
        k0_L,
        x0_R,
        k0_R,
        ms=x_target,
        EQ_L=EQ_L,
        EQ_R=EQ_R,
        k_bound=(float(args.k_min), float(args.k_max)),
    )
    rescue_center_x_raw = float(rescue_center_x_gt)
    rescue_center_x = clamp_to_bounds(rescue_center_x_raw, analysis_xmin, analysis_xmax)
    rescue_k = min(max(float(rescue_k_gt), float(args.k_min)), float(args.k_max))
    gt_used_midpoint = bool(gt_ms_meta.get("used_midpoint_fallback", False))
    if gt_used_midpoint:
        rescue_center_rule = "GT_ms_midpoint_fallback"
        rescue_k_rule = "GT_ms_midpoint_fallback"
    else:
        rescue_center_rule = "GT_ms_target_mean_coordinate"
        rescue_k_rule = "GT_ms_local_harmonic"

    _, _, m_L, sigma_L = EQ_L
    _, _, m_R, sigma_R = EQ_R
    s_eff = float(gt_ms_meta.get("s_eff", float("nan")))

    return {
        "x_rescue_target": float(x_target),
        "target_bin_x": float(target_bin_x),
        "target_bin_index": int(target_bin_index),
        "rescue_center_x_raw": float(rescue_center_x_raw),
        "rescue_center_x": float(rescue_center_x),
        "rescue_center_clamped_to_bounds": bool(abs(rescue_center_x - rescue_center_x_raw) > 0.5 * grid_dx),
        "rescue_center_rule": rescue_center_rule,
        "rescue_k": float(rescue_k),
        "rescue_k_rule": rescue_k_rule,
        "rescue_k_retry_rule": "disabled_for_GT",
        "rescue_k_unclipped": float(rescue_k_gt),
        "rescue_k_saturated": bool(rescue_k >= float(args.k_max) - 1.0e-12),
        "rescue_k_base": float(rescue_k_gt),
        "sigma_target": float("nan"),
        "k_from_sigma": float("nan"),
        "rescue_retry_count": 0,
        "rescue_scale": 1.0,
        "matched_child_name": "",
        "matched_child_side": "",
        "matched_child_center_x": float("nan"),
        "matched_child_target_x": float("nan"),
        "matched_target_distance": float("nan"),
        "matched_child_k": float("nan"),
        "matched_child_raw_k": float("nan"),
        "matched_child_k_rule": "",
        "matched_child_target_source": "",
        "matched_child_used_for_center": 0,
        "gt_left_cluster": gt_left_cluster.name,
        "gt_right_cluster": gt_right_cluster.name,
        "gt_left_boundary": left_boundary.name,
        "gt_right_boundary": right_boundary.name,
        "gt_left_center_x": float(left_boundary.center_x),
        "gt_right_center_x": float(right_boundary.center_x),
        "gt_left_mean_x": float(left_boundary.mean_x),
        "gt_right_mean_x": float(right_boundary.mean_x),
        "gt_left_std_x": float(left_boundary.std_x),
        "gt_right_std_x": float(right_boundary.std_x),
        "gt_left_x_most": float(left_boundary.x_most),
        "gt_right_x_most": float(right_boundary.x_most),
        "gt_m_L": float(m_L),
        "gt_m_R": float(m_R),
        "gt_sigma_L": float(sigma_L),
        "gt_sigma_R": float(sigma_R),
        "gt_s_eff": float(s_eff),
        "gt_x_raw": float(rescue_center_x_raw),
        "gt_k_raw": float(rescue_k_gt),
        "gt_x0_L": float(x0_L),
        "gt_k0_L": float(k0_L),
        "gt_x0_R": float(x0_R),
        "gt_k0_R": float(k0_R),
        "gt_used_midpoint_fallback": bool(gt_used_midpoint),
        "gt_fallback_reason": str(gt_ms_meta.get("fallback_reason", "")),
        "gt_boundary_pair_reason": str(boundary_meta.get("boundary_pair_reason", "")),
        "gt_anchor_coordinate": "mean_x",
    }


def choose_mean_bracketing_windows_for_gt_rescue(
    *,
    target_bin_x: float,
    windows: "list[EnsembleWindow]",
) -> "tuple[EnsembleWindow, EnsembleWindow, dict[str, Any]] | None":
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
                "gt_n_candidate_windows": len(candidates),
            }
    return None


def design_rescue_window_mean_only_gt(
    *,
    target_info: dict[str, Any],
    windows: "list[EnsembleWindow]",
    clusters: "list[EQCluster] | None" = None,
    segment_store: "dict[tuple[str, str], NEQSegment] | None" = None,
    all_segments: "list[NEQSegment] | None" = None,
    neq_patch_store: "dict[str, PMFPatch] | None" = None,
    neq_quad_fit_rows: "list[dict[str, Any]] | None" = None,
    global_pmf: "np.ndarray | None" = None,
    global_variance: "np.ndarray | None" = None,
    grid: np.ndarray,
    args: argparse.Namespace,
    analysis_xmin: float,
    analysis_xmax: float,
    rescue_rows: list[dict[str, Any]] | None = None,
    ctx: dict[str, Any] | None = None,
    rescue_round_root: "Path | None" = None,
) -> dict[str, Any]:
    x_target = float(target_info["target_x"])
    grid_arr = np.asarray(grid, dtype=float)
    grid_dx = abs(float(grid_arr[1] - grid_arr[0])) if len(grid_arr) > 1 else float(args.bin_width)
    target_bin_index = int(np.argmin(np.abs(grid_arr - x_target)))
    target_bin_x = float(grid_arr[target_bin_index])
    kT_ctx = float(ctx["thermal_kT"]) if ctx is not None and "thermal_kT" in ctx else 1.0
    _nan = float("nan")

    mean_pair = choose_mean_bracketing_windows_for_gt_rescue(
        target_bin_x=target_bin_x,
        windows=windows,
    )

    rescue_fit_method = str(getattr(args, "rescue_background_fit_method", "neq"))
    _empty_fit_fields: dict[str, Any] = {
        "rescue_background_fit_method": rescue_fit_method,
        "rescue_fit_source": "", "rescue_fit_segment": "", "rescue_fit_patch": "",
        "rescue_fit_accepted": False, "rescue_fit_fallback_reason": "",
        "rescue_fit_n_bins": 0,
        "rescue_fit_x_min": _nan, "rescue_fit_x_max": _nan,
        "rescue_fit_k0": _nan, "rescue_fit_x0": _nan, "rescue_fit_F0": _nan,
        "rescue_fit_weighted_rmse": _nan, "rescue_fit_reduced_chi2": _nan,
        "rescue_gt_m_L": _nan, "rescue_gt_m_R": _nan,
        "rescue_gt_sigma_L": _nan, "rescue_gt_sigma_R": _nan,
        "rescue_gt_m_target": float(target_bin_x),
        "rescue_gt_s_raw": _nan, "rescue_gt_s_used": _nan,
        "rescue_gt_sigma_target": _nan, "rescue_gt_K_target": _nan,
        "rescue_gt_k_raw": _nan, "rescue_gt_k_final": _nan,
        "rescue_gt_x_raw": _nan, "rescue_gt_x_final": _nan,
        "rescue_gt_x_clipped_to_segment": False,
        "rescue_gt_x_clipped_to_analysis_range": False,
        "rescue_design_rule": "mean_only_GT_fallback_no_bracket",
        # global_fit_* fields (populated when global-pmf fit is attempted)
        "global_fit_method": rescue_fit_method,
        "global_fit_source": "", "global_fit_accepted": False,
        "global_fit_fallback_reason": "",
        "global_fit_n_bins": 0,
        "global_fit_x_min": _nan, "global_fit_x_max": _nan,
        "global_fit_k0": _nan, "global_fit_x0": _nan, "global_fit_F0": _nan,
        "global_fit_a": _nan, "global_fit_b": _nan, "global_fit_c": _nan,
        "global_fit_weighted_rmse": _nan, "global_fit_reduced_chi2": _nan,
        "global_fit_design_rule": "",
        "global_fit_x_res_raw": _nan, "global_fit_x_res": _nan,
        "global_fit_k_res_raw": _nan, "global_fit_k_res": _nan,
        "global_fit_s_raw": _nan, "global_fit_s_used": _nan, "global_fit_sigma_s": _nan,
    }

    if mean_pair is None:
        sigma_fb = max(1.5 * grid_dx, 0.20)
        k_from_sigma = kT_ctx / (sigma_fb ** 2)
        rescue_k_base = max(float(args.k_rescue), k_from_sigma)
        rescue_k = min(max(rescue_k_base, float(args.k_min)), float(args.k_max))
        rescue_cx = clamp_to_bounds(float(target_bin_x), analysis_xmin, analysis_xmax)
        return {
            "x_rescue_target": float(x_target),
            "target_bin_x": float(target_bin_x),
            "target_bin_index": int(target_bin_index),
            "rescue_center_x_raw": float(target_bin_x),
            "rescue_center_x": rescue_cx,
            "rescue_center_clamped_to_bounds": abs(rescue_cx - target_bin_x) > 1e-9,
            "rescue_center_rule": "target_bin_fallback_no_mean_bracket",
            "rescue_k": float(rescue_k),
            "rescue_k_rule": "target_bin_sigma_fallback_no_mean_bracket",
            "rescue_k_retry_rule": "disabled_for_mean_only_GT",
            "rescue_k_unclipped": float(rescue_k_base),
            "rescue_k_saturated": bool(rescue_k >= float(args.k_max) - 1.0e-12),
            "rescue_k_base": float(rescue_k_base),
            "sigma_target": float(sigma_fb),
            "k_from_sigma": float(k_from_sigma),
            "rescue_retry_count": 0,
            "rescue_scale": 1.0,
            "matched_child_name": "",
            "matched_child_side": "",
            "matched_child_center_x": _nan,
            "matched_child_target_x": _nan,
            "matched_target_distance": _nan,
            "matched_child_k": _nan,
            "matched_child_raw_k": _nan,
            "matched_child_k_rule": "",
            "matched_child_target_source": "",
            "matched_child_used_for_center": 0,
            "gt_pair_rule": "no_adjacent_mean_bracketing_pair",
            "gt_left_cluster": "",
            "gt_right_cluster": "",
            "gt_left_boundary": "",
            "gt_right_boundary": "",
            "gt_left_center_x": _nan,
            "gt_right_center_x": _nan,
            "gt_left_mean_x": _nan,
            "gt_right_mean_x": _nan,
            "gt_left_std_x": _nan,
            "gt_right_std_x": _nan,
            "gt_left_x_most": _nan,
            "gt_right_x_most": _nan,
            "gt_m_L": _nan,
            "gt_m_R": _nan,
            "gt_sigma_L": _nan,
            "gt_sigma_R": _nan,
            "gt_s_raw": _nan,
            "gt_s_used": _nan,
            "gt_s_fallback_to_midpoint": False,
            "gt_s_eff": _nan,
            "gt_x_raw": _nan,
            "gt_k_raw": _nan,
            "gt_x_clipped": False,
            "gt_k_clipped": False,
            "gt_x0_L": _nan,
            "gt_k0_L": _nan,
            "gt_x0_R": _nan,
            "gt_k0_R": _nan,
            "gt_used_midpoint_fallback": False,
            "gt_fallback_reason": "no_adjacent_mean_bracketing_pair",
            "gt_boundary_pair_reason": "",
            "gt_anchor_coordinate": "mean_x",
            "mo_rescue_design_method": "mean_only_background_sigma_gt_width",
            "mo_left_window": "",
            "mo_right_window": "",
            "mo_x_L": _nan, "mo_k_L": _nan, "mo_m_L": _nan, "mo_sigma_L": _nan,
            "mo_x_R": _nan, "mo_k_R": _nan, "mo_m_R": _nan, "mo_sigma_R": _nan,
            "mo_x_target": float(target_bin_x),
            "mo_s_raw": _nan, "mo_s_used": _nan, "mo_s_fallback_to_midpoint": False,
            "mo_sigma_s": _nan, "mo_sigma_s_fallback_used": False,
            "mo_k0_mean_only": _nan, "mo_x0_mean_only": _nan,
            "mo_x0_right_check": _nan, "mo_x0_left_right_abs_diff": _nan,
            "mo_kT_eff_L": _nan, "mo_kT_eff_R": _nan, "mo_kT_eff": _nan,
            "mo_kT_eff_ratio": _nan, "mo_kT_eff_fallback_used": False,
            "mo_k_res_raw": _nan, "mo_k_res": float(rescue_k), "mo_k_res_clipped_to": "",
            "mo_x_res_raw": float(target_bin_x), "mo_x_res": rescue_cx,
            "mo_x_res_clipped": abs(rescue_cx - target_bin_x) > 1e-9,
            "mo_fallback_reason": "no_adjacent_mean_bracketing_pair",
            **_empty_fit_fields,
        }

    left_boundary, right_boundary, pair_meta = mean_pair

    x_L = float(left_boundary.center_x)
    k_L = float(left_boundary.k)
    m_L = float(left_boundary.mean_x)
    sigma_L = float(left_boundary.std_x)

    x_R = float(right_boundary.center_x)
    k_R = float(right_boundary.k)
    m_R = float(right_boundary.mean_x)
    sigma_R = float(right_boundary.std_x)

    # Old GT harmonic params kept for backward compat with existing notebook cells.
    EQ_L = eq_gt_tuple(left_boundary)
    EQ_R = eq_gt_tuple(right_boundary)
    x0_L_gt, k0_L_gt, x0_R_gt, k0_R_gt, _hm = get_k0_x0_harmonic_fromEQ(EQ_L, EQ_R)

    # ---- Global PMF quadratic fit (when rescue_background_fit_method is global-pmf or auto) ----
    if rescue_fit_method in ("global-pmf", "auto") and global_pmf is not None:
        _var_floor = float(getattr(args, "variance_floor", 1.0e-6))
        _min_bins = int(getattr(args, "rescue_global_fit_min_bins",
                                getattr(args, "rescue_neq_fit_min_bins", 5)))
        _k0_min = float(getattr(args, "rescue_global_fit_k0_min_abs",
                                getattr(args, "rescue_neq_fit_k0_min_abs", 1.0e-8)))
        _x0_mf = float(getattr(args, "rescue_global_fit_x0_margin_factor",
                               getattr(args, "rescue_neq_fit_x0_margin_factor", 0.25)))
        _fit_radius = getattr(args, "rescue_global_fit_radius", None)
        gfit = fit_quadratic_background_from_global_pmf(
            target_bin_x=target_bin_x,
            grid=grid,
            global_pmf=global_pmf,
            global_variance=global_variance,
            variance_floor=_var_floor,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            min_fit_bins=_min_bins,
            k0_min_abs=_k0_min,
            x0_margin_factor=_x0_mf,
            bin_width=float(grid_dx),
            fit_radius=_fit_radius,
        )
        # Always record global fit diagnostics
        _empty_fit_fields.update({
            "global_fit_method": rescue_fit_method,
            "global_fit_source": str(gfit.get("fit_source", "")),
            "global_fit_accepted": bool(gfit.get("fit_accepted", False)),
            "global_fit_fallback_reason": str(gfit.get("fallback_reason", "")),
            "global_fit_n_bins": int(gfit.get("n_fit_bins", 0)),
            "global_fit_x_min": float(gfit.get("x_fit_min", _nan)),
            "global_fit_x_max": float(gfit.get("x_fit_max", _nan)),
            "global_fit_k0": float(gfit.get("k0", _nan)),
            "global_fit_x0": float(gfit.get("x0", _nan)),
            "global_fit_F0": float(gfit.get("F0", _nan)),
            "global_fit_a": float(gfit.get("a", _nan)),
            "global_fit_b": float(gfit.get("b", _nan)),
            "global_fit_c": float(gfit.get("c", _nan)),
            "global_fit_weighted_rmse": float(gfit.get("weighted_rmse", _nan)),
            "global_fit_reduced_chi2": float(gfit.get("reduced_chi2", _nan)),
        })

        if gfit["fit_accepted"]:
            k0_gf = float(gfit["k0"])
            x0_gf = float(gfit["x0"])
            # Interpolation coordinate
            _denom_gf = m_R - m_L
            if abs(_denom_gf) < 1e-9:
                _s_raw_gf = _nan
                _s_used_gf = 0.5
                _s_fb_gf = True
            else:
                _s_raw_gf = (target_bin_x - m_L) / _denom_gf
                if 0.0 < _s_raw_gf < 1.0:
                    _s_used_gf = _s_raw_gf
                    _s_fb_gf = False
                else:
                    _s_used_gf = 0.5
                    _s_fb_gf = True
            _sigma_gf = (1.0 - _s_used_gf) * sigma_L + _s_used_gf * sigma_R
            if _sigma_gf <= 0.0 or not np.isfinite(_sigma_gf):
                gfit["fit_accepted"] = False
                gfit["fallback_reason"] = (gfit.get("fallback_reason") or "") + ":invalid_sigma_target"
            else:
                _K_gf = 1.0 / (_sigma_gf ** 2)
                _k_raw_gf = _K_gf - k0_gf
                _k_final_gf = float(np.clip(_k_raw_gf, float(args.k_min), float(args.k_max)))
                _k_clipped_gf = (
                    "k_min" if _k_raw_gf < float(args.k_min)
                    else ("k_max" if _k_raw_gf > float(args.k_max) else "")
                )
                if not np.isfinite(_k_final_gf) or _k_final_gf <= 0.0:
                    gfit["fit_accepted"] = False
                    gfit["fallback_reason"] = (gfit.get("fallback_reason") or "") + ":invalid_k_rescue"
                else:
                    _x_raw_gf = ((k0_gf + _k_final_gf) * target_bin_x - k0_gf * x0_gf) / _k_final_gf
                    if not np.isfinite(_x_raw_gf):
                        gfit["fit_accepted"] = False
                        gfit["fallback_reason"] = (gfit.get("fallback_reason") or "") + ":nonfinite_x_rescue"
                    else:
                        _seg_lo = min(float(left_boundary.center_x), float(right_boundary.center_x))
                        _seg_hi = max(float(left_boundary.center_x), float(right_boundary.center_x))
                        _x_clamp1_gf = float(np.clip(_x_raw_gf, _seg_lo, _seg_hi))
                        _clipped_seg_gf = abs(_x_clamp1_gf - _x_raw_gf) > 1e-9
                        _x_final_gf = float(np.clip(_x_clamp1_gf, analysis_xmin, analysis_xmax))
                        _clipped_ana_gf = abs(_x_final_gf - _x_clamp1_gf) > 1e-9
                        _gfit_fields: dict[str, Any] = {
                            "rescue_background_fit_method": rescue_fit_method,
                            "rescue_fit_source": "global_pmf_quadratic_fit",
                            "rescue_fit_segment": "",
                            "rescue_fit_patch": "global_pmf",
                            "rescue_fit_accepted": True,
                            "rescue_fit_fallback_reason": "",
                            "rescue_fit_n_bins": int(gfit.get("n_fit_bins", 0)),
                            "rescue_fit_x_min": float(gfit.get("x_fit_min", _nan)),
                            "rescue_fit_x_max": float(gfit.get("x_fit_max", _nan)),
                            "rescue_fit_k0": k0_gf,
                            "rescue_fit_x0": x0_gf,
                            "rescue_fit_F0": float(gfit.get("F0", _nan)),
                            "rescue_fit_weighted_rmse": float(gfit.get("weighted_rmse", _nan)),
                            "rescue_fit_reduced_chi2": float(gfit.get("reduced_chi2", _nan)),
                            "rescue_gt_m_L": m_L, "rescue_gt_m_R": m_R,
                            "rescue_gt_sigma_L": sigma_L, "rescue_gt_sigma_R": sigma_R,
                            "rescue_gt_m_target": float(target_bin_x),
                            "rescue_gt_s_raw": float(_s_raw_gf),
                            "rescue_gt_s_used": float(_s_used_gf),
                            "rescue_gt_sigma_target": float(_sigma_gf),
                            "rescue_gt_K_target": float(_K_gf),
                            "rescue_gt_k_raw": float(_k_raw_gf),
                            "rescue_gt_k_final": float(_k_final_gf),
                            "rescue_gt_x_raw": float(_x_raw_gf),
                            "rescue_gt_x_final": float(_x_final_gf),
                            "rescue_gt_x_clipped_to_segment": bool(_clipped_seg_gf),
                            "rescue_gt_x_clipped_to_analysis_range": bool(_clipped_ana_gf),
                            "rescue_design_rule": "global_pmf_quadratic_fit_GT",
                            "global_fit_design_rule": "global_pmf_quadratic_fit_GT",
                            "global_fit_x_res_raw": float(_x_raw_gf),
                            "global_fit_x_res": float(_x_final_gf),
                            "global_fit_k_res_raw": float(_k_raw_gf),
                            "global_fit_k_res": float(_k_final_gf),
                            "global_fit_s_raw": float(_s_raw_gf),
                            "global_fit_s_used": float(_s_used_gf),
                            "global_fit_sigma_s": float(_sigma_gf),
                        }
                        return {
                            "x_rescue_target": float(x_target),
                            "target_bin_x": float(target_bin_x),
                            "target_bin_index": int(target_bin_index),
                            "rescue_center_x_raw": float(_x_raw_gf),
                            "rescue_center_x": float(_x_final_gf),
                            "rescue_center_clamped_to_bounds": bool(_clipped_seg_gf or _clipped_ana_gf),
                            "rescue_center_rule": "global_pmf_quadratic_fit_GT",
                            "rescue_k": float(_k_final_gf),
                            "rescue_k_rule": "global_pmf_fit_K_target_minus_k0",
                            "rescue_k_retry_rule": "disabled_for_global_pmf_GT",
                            "rescue_k_unclipped": float(_k_raw_gf),
                            "rescue_k_saturated": bool(_k_final_gf >= float(args.k_max) - 1.0e-12),
                            "rescue_k_base": float(_k_raw_gf),
                            "sigma_target": float(_sigma_gf),
                            "k_from_sigma": float(_K_gf),
                            "rescue_retry_count": 0,
                            "rescue_scale": 1.0,
                            "matched_child_name": "",
                            "matched_child_side": "",
                            "matched_child_center_x": _nan,
                            "matched_child_target_x": _nan,
                            "matched_target_distance": _nan,
                            "matched_child_k": _nan,
                            "matched_child_raw_k": _nan,
                            "matched_child_k_rule": "",
                            "matched_child_target_source": "",
                            "matched_child_used_for_center": 0,
                            "gt_pair_rule": str(pair_meta.get("gt_pair_rule", "")),
                            "gt_left_cluster": "",
                            "gt_right_cluster": "",
                            "gt_left_boundary": left_boundary.name,
                            "gt_right_boundary": right_boundary.name,
                            "gt_left_center_x": float(left_boundary.center_x),
                            "gt_right_center_x": float(right_boundary.center_x),
                            "gt_left_mean_x": float(left_boundary.mean_x),
                            "gt_right_mean_x": float(right_boundary.mean_x),
                            "gt_left_std_x": float(left_boundary.std_x),
                            "gt_right_std_x": float(right_boundary.std_x),
                            "gt_left_x_most": float(left_boundary.x_most),
                            "gt_right_x_most": float(right_boundary.x_most),
                            "gt_m_L": m_L, "gt_m_R": m_R,
                            "gt_sigma_L": sigma_L, "gt_sigma_R": sigma_R,
                            "gt_s_raw": float(_s_raw_gf),
                            "gt_s_used": float(_s_used_gf),
                            "gt_s_fallback_to_midpoint": bool(_s_fb_gf),
                            "gt_s_eff": float(_s_used_gf),
                            "gt_x_raw": float(_x_raw_gf),
                            "gt_k_raw": float(_k_raw_gf),
                            "gt_x_clipped": bool(_clipped_seg_gf or _clipped_ana_gf),
                            "gt_k_clipped": bool(_k_clipped_gf != ""),
                            "gt_x0_L": float(x0_L_gt),
                            "gt_k0_L": float(k0_L_gt),
                            "gt_x0_R": float(x0_R_gt),
                            "gt_k0_R": float(k0_R_gt),
                            "gt_used_midpoint_fallback": bool(_s_fb_gf),
                            "gt_fallback_reason": "",
                            "gt_boundary_pair_reason": "",
                            "gt_anchor_coordinate": "mean_x",
                            "mo_rescue_design_method": "global_pmf_quadratic_fit_GT",
                            "mo_left_window": left_boundary.name,
                            "mo_right_window": right_boundary.name,
                            "mo_x_L": float(left_boundary.center_x),
                            "mo_k_L": float(left_boundary.k),
                            "mo_m_L": m_L, "mo_sigma_L": sigma_L,
                            "mo_x_R": float(right_boundary.center_x),
                            "mo_k_R": float(right_boundary.k),
                            "mo_m_R": m_R, "mo_sigma_R": sigma_R,
                            "mo_x_target": float(target_bin_x),
                            "mo_s_raw": float(_s_raw_gf), "mo_s_used": float(_s_used_gf),
                            "mo_s_fallback_to_midpoint": bool(_s_fb_gf),
                            "mo_sigma_s": float(_sigma_gf), "mo_sigma_s_fallback_used": False,
                            "mo_k0_mean_only": k0_gf, "mo_x0_mean_only": x0_gf,
                            "mo_x0_right_check": _nan, "mo_x0_left_right_abs_diff": _nan,
                            "mo_kT_eff_L": _nan, "mo_kT_eff_R": _nan,
                            "mo_kT_eff": _nan, "mo_kT_eff_ratio": _nan,
                            "mo_kT_eff_fallback_used": False,
                            "mo_k_res_raw": float(_k_raw_gf),
                            "mo_k_res": float(_k_final_gf),
                            "mo_k_res_clipped_to": _k_clipped_gf,
                            "mo_x_res_raw": float(_x_raw_gf),
                            "mo_x_res": float(_x_final_gf),
                            "mo_x_res_clipped": bool(_clipped_seg_gf or _clipped_ana_gf),
                            "mo_fallback_reason": "",
                            **{**_empty_fit_fields, **_gfit_fields},
                        }
        # Global PMF fit not accepted — update empty_fit_fields with fallback info
        _empty_fit_fields.update({
            "rescue_background_fit_method": rescue_fit_method,
            "rescue_fit_source": str(gfit.get("fit_source", "")),
            "rescue_fit_segment": "",
            "rescue_fit_patch": "global_pmf",
            "rescue_fit_accepted": False,
            "rescue_fit_fallback_reason": str(gfit.get("fallback_reason", "")),
            "rescue_fit_n_bins": int(gfit.get("n_fit_bins", 0)),
            "rescue_fit_k0": float(gfit.get("k0", _nan)),
            "rescue_fit_x0": float(gfit.get("x0", _nan)),
            "rescue_design_rule": "mean_only_GT_fallback_global_pmf_fit_rejected",
            "global_fit_design_rule": "rejected",
        })

    # ---- mean-only diagnostic variables (initialized to nan/default) ----
    fallback_reason = ""
    s_raw: float = _nan
    s_used: float = _nan
    s_fallback = False
    sigma_s: float = _nan
    sigma_s_fallback_used = False
    k0: float = _nan
    x0: float = _nan
    x0_right_check: float = _nan
    x0_abs_diff: float = _nan
    kT_eff_L: float = _nan
    kT_eff_R: float = _nan
    kT_eff: float = _nan
    kT_eff_ratio: float = _nan
    kT_eff_fallback_used = False
    k_res_raw_val: float = _nan
    k_res: float = _nan
    k_res_clipped_to = ""
    x_res_raw: float = _nan
    x_res: float = _nan
    x_res_clipped = False

    denom = m_R - m_L

    if abs(denom) < 1e-9:
        # ---- degenerate-means fallback ----
        fallback_reason = "degenerate_neighbor_means"
        s_used = 0.5
        s_fallback = True
        sigma_s_cand = 0.5 * (abs(sigma_L) + abs(sigma_R))
        if sigma_s_cand <= 0.0 or not np.isfinite(sigma_s_cand):
            sigma_s = max(float(args.bin_width), 1e-6)
            sigma_s_fallback_used = True
        else:
            sigma_s = sigma_s_cand
        kT_eff_fallback_used = True
        k_res_raw_val = max(float(args.k_rescue), float(args.k_min))
        k_res = float(np.clip(k_res_raw_val, float(args.k_min), float(args.k_max)))
        if k_res_raw_val < float(args.k_min):
            k_res_clipped_to = "k_min"
        elif k_res_raw_val > float(args.k_max):
            k_res_clipped_to = "k_max"
        x_res_raw = target_bin_x
        x_res = float(np.clip(x_res_raw, analysis_xmin, analysis_xmax))
        x_res_clipped = abs(x_res - x_res_raw) > 1e-9
    else:
        # ---- mean-only force-balance ----
        k0 = (k_L * (m_L - x_L) - k_R * (m_R - x_R)) / denom
        near_zero_k0 = abs(k0) < 1e-12
        if near_zero_k0:
            fallback_reason = "near_zero_k0_center_shift_disabled"
        else:
            x0 = m_L + (k_L / k0) * (m_L - x_L)
            x0_right_check = m_R + (k_R / k0) * (m_R - x_R)
            x0_abs_diff = abs(x0_right_check - x0)

        # ---- interpolation coordinate ----
        s_raw = (target_bin_x - m_L) / denom
        if 0.0 < s_raw < 1.0:
            s_used = s_raw
        else:
            s_used = 0.5
            s_fallback = True

        # ---- target width ----
        sigma_s_cand = (1.0 - s_used) * sigma_L + s_used * sigma_R
        if sigma_s_cand <= 0.0 or not np.isfinite(sigma_s_cand):
            sigma_s = max(0.5 * (abs(sigma_L) + abs(sigma_R)), float(args.bin_width), 1e-6)
            sigma_s_fallback_used = True
        else:
            sigma_s = sigma_s_cand

        # ---- self-calibrated kT_eff ----
        kT_eff_L = (k0 + k_L) * sigma_L ** 2
        kT_eff_R = (k0 + k_R) * sigma_R ** 2
        kT_eff_tentative: float = _nan
        if np.isfinite(kT_eff_L) and np.isfinite(kT_eff_R):
            kT_eff_tentative = 0.5 * (kT_eff_L + kT_eff_R)
            if kT_eff_L > 0.0 and kT_eff_R > 0.0:
                kT_eff_ratio = max(kT_eff_L, kT_eff_R) / min(kT_eff_L, kT_eff_R)
        if kT_eff_tentative <= 0.0 or not np.isfinite(kT_eff_tentative):
            kT_eff = kT_ctx
            kT_eff_fallback_used = True
        else:
            kT_eff = kT_eff_tentative

        # ---- rescue stiffness ----
        if sigma_s > 0.0 and np.isfinite(sigma_s) and np.isfinite(kT_eff) and np.isfinite(k0):
            k_res_raw_val = kT_eff / (sigma_s ** 2) - k0
        else:
            k_res_raw_val = max(float(args.k_rescue), float(args.k_min))
            if not fallback_reason:
                fallback_reason = "invalid_sigma_or_kT_eff_for_k_res"
        k_res = float(np.clip(k_res_raw_val, float(args.k_min), float(args.k_max)))
        if k_res_raw_val < float(args.k_min):
            k_res_clipped_to = "k_min"
        elif k_res_raw_val > float(args.k_max):
            k_res_clipped_to = "k_max"

        # ---- rescue center (signed k0, no abs) ----
        if near_zero_k0 or not np.isfinite(x0) or not np.isfinite(k_res) or k_res == 0.0:
            x_res_raw = target_bin_x
        else:
            x_res_raw = target_bin_x + (k0 / k_res) * (target_bin_x - x0)
        x_res = float(np.clip(x_res_raw, analysis_xmin, analysis_xmax))
        x_res_clipped = abs(x_res - x_res_raw) > 1e-9

    rescue_cx_final = x_res if np.isfinite(x_res) else clamp_to_bounds(float(target_bin_x), analysis_xmin, analysis_xmax)
    rescue_k_final = k_res if np.isfinite(k_res) else max(float(args.k_rescue), float(args.k_min))

    return {
        "x_rescue_target": float(x_target),
        "target_bin_x": float(target_bin_x),
        "target_bin_index": int(target_bin_index),
        "rescue_center_x_raw": float(x_res_raw) if np.isfinite(x_res_raw) else float(target_bin_x),
        "rescue_center_x": float(rescue_cx_final),
        "rescue_center_clamped_to_bounds": bool(x_res_clipped),
        "rescue_center_rule": (
            "mean_only_background_sigma_gt_width"
            if not fallback_reason
            else f"mean_only_GT_fallback_{fallback_reason}"
        ),
        "rescue_k": float(rescue_k_final),
        "rescue_k_rule": "mean_only_kT_eff_over_sigma_s_sq_minus_k0",
        "rescue_k_retry_rule": "disabled_for_mean_only_GT",
        "rescue_k_unclipped": float(k_res_raw_val) if np.isfinite(k_res_raw_val) else _nan,
        "rescue_k_saturated": bool(float(rescue_k_final) >= float(args.k_max) - 1.0e-12),
        "rescue_k_base": float(k_res_raw_val) if np.isfinite(k_res_raw_val) else _nan,
        "sigma_target": float(sigma_s) if np.isfinite(sigma_s) else _nan,
        "k_from_sigma": _nan,
        "rescue_retry_count": 0,
        "rescue_scale": 1.0,
        "matched_child_name": "",
        "matched_child_side": "",
        "matched_child_center_x": _nan,
        "matched_child_target_x": _nan,
        "matched_target_distance": _nan,
        "matched_child_k": _nan,
        "matched_child_raw_k": _nan,
        "matched_child_k_rule": "",
        "matched_child_target_source": "",
        "matched_child_used_for_center": 0,
        "gt_pair_rule": str(pair_meta.get("gt_pair_rule", "")),
        "gt_left_cluster": "",
        "gt_right_cluster": "",
        "gt_left_boundary": left_boundary.name,
        "gt_right_boundary": right_boundary.name,
        "gt_left_center_x": float(left_boundary.center_x),
        "gt_right_center_x": float(right_boundary.center_x),
        "gt_left_mean_x": float(left_boundary.mean_x),
        "gt_right_mean_x": float(right_boundary.mean_x),
        "gt_left_std_x": float(left_boundary.std_x),
        "gt_right_std_x": float(right_boundary.std_x),
        "gt_left_x_most": float(left_boundary.x_most),
        "gt_right_x_most": float(right_boundary.x_most),
        "gt_m_L": m_L,
        "gt_m_R": m_R,
        "gt_sigma_L": sigma_L,
        "gt_sigma_R": sigma_R,
        "gt_s_raw": float(s_raw),
        "gt_s_used": float(s_used) if np.isfinite(s_used) else _nan,
        "gt_s_fallback_to_midpoint": bool(s_fallback),
        "gt_s_eff": float(s_used) if np.isfinite(s_used) else _nan,
        "gt_x_raw": float(x_res_raw) if np.isfinite(x_res_raw) else _nan,
        "gt_k_raw": float(k_res_raw_val) if np.isfinite(k_res_raw_val) else _nan,
        "gt_x_clipped": bool(x_res_clipped),
        "gt_k_clipped": bool(k_res_clipped_to != ""),
        "gt_x0_L": float(x0_L_gt),
        "gt_k0_L": float(k0_L_gt),
        "gt_x0_R": float(x0_R_gt),
        "gt_k0_R": float(k0_R_gt),
        "gt_used_midpoint_fallback": bool(s_fallback),
        "gt_fallback_reason": str(fallback_reason),
        "gt_boundary_pair_reason": "",
        "gt_anchor_coordinate": "mean_x",
        # ---- mean-only diagnostics ----
        "mo_rescue_design_method": "mean_only_background_sigma_gt_width",
        "mo_left_window": left_boundary.name,
        "mo_right_window": right_boundary.name,
        "mo_x_L": x_L, "mo_k_L": k_L, "mo_m_L": m_L, "mo_sigma_L": sigma_L,
        "mo_x_R": x_R, "mo_k_R": k_R, "mo_m_R": m_R, "mo_sigma_R": sigma_R,
        "mo_x_target": float(target_bin_x),
        "mo_s_raw": float(s_raw),
        "mo_s_used": float(s_used),
        "mo_s_fallback_to_midpoint": bool(s_fallback),
        "mo_sigma_s": float(sigma_s),
        "mo_sigma_s_fallback_used": bool(sigma_s_fallback_used),
        "mo_k0_mean_only": float(k0),
        "mo_x0_mean_only": float(x0),
        "mo_x0_right_check": float(x0_right_check),
        "mo_x0_left_right_abs_diff": float(x0_abs_diff),
        "mo_kT_eff_L": float(kT_eff_L),
        "mo_kT_eff_R": float(kT_eff_R),
        "mo_kT_eff": float(kT_eff),
        "mo_kT_eff_ratio": float(kT_eff_ratio),
        "mo_kT_eff_fallback_used": bool(kT_eff_fallback_used),
        "mo_k_res_raw": float(k_res_raw_val),
        "mo_k_res": float(k_res),
        "mo_k_res_clipped_to": str(k_res_clipped_to),
        "mo_x_res_raw": float(x_res_raw),
        "mo_x_res": float(x_res),
        "mo_x_res_clipped": bool(x_res_clipped),
        "mo_fallback_reason": str(fallback_reason),
        **{
            **_empty_fit_fields,
            "rescue_gt_m_L": m_L, "rescue_gt_m_R": m_R,
            "rescue_gt_sigma_L": sigma_L, "rescue_gt_sigma_R": sigma_R,
            "rescue_gt_m_target": float(target_bin_x),
            "rescue_gt_s_raw": float(s_raw),
            "rescue_gt_s_used": float(s_used) if np.isfinite(s_used) else _nan,
            "rescue_gt_sigma_target": float(sigma_s) if np.isfinite(sigma_s) else _nan,
            "rescue_gt_K_target": float(kT_eff / sigma_s ** 2) if np.isfinite(sigma_s) and sigma_s > 0 and np.isfinite(kT_eff) else _nan,
            "rescue_gt_k_raw": float(k_res_raw_val) if np.isfinite(k_res_raw_val) else _nan,
            "rescue_gt_k_final": float(rescue_k_final),
            "rescue_gt_x_raw": float(x_res_raw) if np.isfinite(x_res_raw) else _nan,
            "rescue_gt_x_final": float(rescue_cx_final),
            "rescue_gt_x_clipped_to_segment": False,
            "rescue_gt_x_clipped_to_analysis_range": bool(x_res_clipped),
            "rescue_design_rule": (
                _empty_fit_fields["rescue_design_rule"]
                if _empty_fit_fields["rescue_design_rule"] != "mean_only_GT_fallback_no_bracket"
                else "mean_only_background_sigma_gt_width"
            ),
        },
    }


def gap_clusters_for_target(
    target_x: float,
    clusters: list[EQCluster],
) -> tuple[EQCluster, EQCluster] | None:
    for left_cluster, right_cluster in zip(clusters[:-1], clusters[1:]):
        if float(left_cluster.right_x) + 1.0e-9 < float(target_x) < float(right_cluster.left_x) - 1.0e-9:
            return left_cluster, right_cluster
    return None


def cluster_covering_target(target_x: float, clusters: list[EQCluster]) -> EQCluster | None:
    for cluster in clusters:
        if float(cluster.left_x) - 1.0e-9 <= float(target_x) <= float(cluster.right_x) + 1.0e-9:
            return cluster
    return None


def segment_covering_target(target_x: float, segments: list[NEQSegment]) -> NEQSegment | None:
    """Return a segment whose mean_x boundary interval brackets target_x."""
    for segment in segments:
        left_x = float(segment.left_boundary.mean_x)
        right_x = float(segment.right_boundary.mean_x)
        lo = min(left_x, right_x)
        hi = max(left_x, right_x)
        if lo - 1.0e-9 <= float(target_x) <= hi + 1.0e-9:
            return segment
    return None


def disconnected_segments(segments: list[NEQSegment]) -> list[NEQSegment]:
    return [
        segment
        for segment in segments
        if segment.connectivity and not bool(segment.connectivity.get("eop_crossed", True))
    ]


def mts_failed_segments(segments: list[NEQSegment]) -> list[NEQSegment]:
    return [
        segment
        for segment in segments
        if (segment.neq_patch_decision or segment.mts_patch_built or segment.cft_summary)
        and not bool(segment.mts_patch_built)
    ]


def find_first_rescue_pair(
    clusters: list[EQCluster],
    windows: list[EnsembleWindow],
) -> dict[str, Any] | None:
    rescue_windows = [w for w in windows if str(w.side) == "rescue"]
    for left_cluster, right_cluster in zip(clusters[:-1], clusters[1:]):
        left_boundary = rightmost_mean_window(left_cluster)
        right_boundary = leftmost_mean_window(right_cluster)
        lo = min(float(left_boundary.mean_x), float(right_boundary.mean_x))
        hi = max(float(left_boundary.mean_x), float(right_boundary.mean_x))
        has_existing_rescue = any(
            lo < float(w.mean_x) < hi
            for w in rescue_windows
        )
        if not has_existing_rescue:
            return {
                "left_cluster": left_cluster,
                "right_cluster": right_cluster,
                "left_boundary": left_boundary,
                "right_boundary": right_boundary,
            }
    return None


def budget_exhausted_for_future_sampling(args: argparse.Namespace, budget: BudgetTracker) -> bool:
    next_growth_cost = stage_cost_neq(int(args.n_neq_traj), int(args.t_neq)) + 2 * stage_cost_eq(int(args.n_eq_steps))
    next_rescue_cost = stage_cost_eq(int(args.n_eq_steps))
    min_required = min(next_growth_cost, next_rescue_cost)
    return not budget.can_spend(int(min_required))


def summarize_timing_rows(timing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in timing_rows:
        groups.setdefault(str(row.get("operation", "")), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for operation, rows in sorted(groups.items()):
        wall = [float(row.get("wall_seconds", 0.0) or 0.0) for row in rows]
        cpu = [float(row.get("cpu_seconds", 0.0) or 0.0) for row in rows]
        summary_rows.append({
            "operation": operation,
            "count": int(len(rows)),
            "total_wall_seconds": float(sum(wall)),
            "mean_wall_seconds": float(sum(wall) / len(wall)) if wall else 0.0,
            "max_wall_seconds": float(max(wall)) if wall else 0.0,
            "total_cpu_seconds": float(sum(cpu)),
            "mean_cpu_seconds": float(sum(cpu) / len(cpu)) if cpu else 0.0,
            "max_cpu_seconds": float(max(cpu)) if cpu else 0.0,
            "n_error": int(sum(1 for row in rows if row.get("status") == "error")),
        })
    return summary_rows


def write_timing_outputs(out_root: Path, timing_rows: list[dict[str, Any]]) -> None:
    write_csv(
        out_root / "operation_timing.csv",
        ordered_fieldnames(
            timing_rows,
            extras=[
                "stage", "operation", "item",
                "wall_seconds", "cpu_seconds", "status", "error",
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
                "operation", "count",
                "total_wall_seconds", "mean_wall_seconds", "max_wall_seconds",
                "total_cpu_seconds", "mean_cpu_seconds", "max_cpu_seconds",
                "n_error",
            ],
        ),
        summary_rows,
    )


_RESCUE_SUMMARY_COLS = [
    "round",
    "target_priority",
    "target_reason",
    "x_rescue_target",
    "target_bin_x",
    "target_variance",
    "uncovered_start_x",
    "uncovered_end_x",
    "uncovered_width",
    "uncovered_n_bins",
    "rescue_center_x_raw",
    "rescue_center_x",
    "rescue_center_clamped_to_bounds",
    "rescue_center_rule",
    "sigma_target",
    "k_from_sigma",
    "rescue_k_base",
    "rescue_k",
    "rescue_k_unclipped",
    "rescue_k_saturated",
    "rescue_k_rule",
    "rescue_k_retry_rule",
    "matched_child_name",
    "matched_child_side",
    "matched_child_center_x",
    "matched_child_target_x",
    "matched_target_distance",
    "matched_child_k",
    "matched_child_raw_k",
    "matched_child_k_rule",
    "matched_child_target_source",
    "matched_child_used_for_center",
    "rescue_retry_count",
    "rescue_scale",
    "gt_pair_rule",
    "gt_left_cluster",
    "gt_right_cluster",
    "gt_left_boundary",
    "gt_right_boundary",
    "gt_left_center_x",
    "gt_right_center_x",
    "gt_left_mean_x",
    "gt_right_mean_x",
    "gt_left_std_x",
    "gt_right_std_x",
    "gt_left_x_most",
    "gt_right_x_most",
    "gt_m_L",
    "gt_m_R",
    "gt_sigma_L",
    "gt_sigma_R",
    "gt_s_raw",
    "gt_s_used",
    "gt_s_fallback_to_midpoint",
    "gt_s_eff",
    "gt_x_raw",
    "gt_k_raw",
    "gt_x_clipped",
    "gt_k_clipped",
    "gt_x0_L",
    "gt_k0_L",
    "gt_x0_R",
    "gt_k0_R",
    "gt_used_midpoint_fallback",
    "gt_fallback_reason",
    "gt_boundary_pair_reason",
    "gt_anchor_coordinate",
    "mo_rescue_design_method",
    "mo_left_window",
    "mo_right_window",
    "mo_x_L",
    "mo_k_L",
    "mo_m_L",
    "mo_sigma_L",
    "mo_x_R",
    "mo_k_R",
    "mo_m_R",
    "mo_sigma_R",
    "mo_x_target",
    "mo_s_raw",
    "mo_s_used",
    "mo_s_fallback_to_midpoint",
    "mo_sigma_s",
    "mo_sigma_s_fallback_used",
    "mo_k0_mean_only",
    "mo_x0_mean_only",
    "mo_x0_right_check",
    "mo_x0_left_right_abs_diff",
    "mo_kT_eff_L",
    "mo_kT_eff_R",
    "mo_kT_eff",
    "mo_kT_eff_ratio",
    "mo_kT_eff_fallback_used",
    "mo_k_res_raw",
    "mo_k_res",
    "mo_k_res_clipped_to",
    "mo_x_res_raw",
    "mo_x_res",
    "mo_x_res_clipped",
    "mo_fallback_reason",
    "rescue_background_fit_method",
    "rescue_fit_source",
    "rescue_fit_segment",
    "rescue_fit_patch",
    "rescue_fit_accepted",
    "rescue_fit_fallback_reason",
    "rescue_fit_n_bins",
    "rescue_fit_x_min",
    "rescue_fit_x_max",
    "rescue_fit_k0",
    "rescue_fit_x0",
    "rescue_fit_F0",
    "rescue_fit_weighted_rmse",
    "rescue_fit_reduced_chi2",
    "rescue_gt_m_L",
    "rescue_gt_m_R",
    "rescue_gt_sigma_L",
    "rescue_gt_sigma_R",
    "rescue_gt_m_target",
    "rescue_gt_s_raw",
    "rescue_gt_s_used",
    "rescue_gt_sigma_target",
    "rescue_gt_K_target",
    "rescue_gt_k_raw",
    "rescue_gt_k_final",
    "rescue_gt_x_raw",
    "rescue_gt_x_final",
    "rescue_gt_x_clipped_to_segment",
    "rescue_gt_x_clipped_to_analysis_range",
    "rescue_design_rule",
    "global_fit_method",
    "global_fit_source",
    "global_fit_accepted",
    "global_fit_fallback_reason",
    "global_fit_n_bins",
    "global_fit_x_min",
    "global_fit_x_max",
    "global_fit_k0",
    "global_fit_x0",
    "global_fit_F0",
    "global_fit_a",
    "global_fit_b",
    "global_fit_c",
    "global_fit_weighted_rmse",
    "global_fit_reduced_chi2",
    "global_fit_design_rule",
    "global_fit_x_res_raw",
    "global_fit_x_res",
    "global_fit_k_res_raw",
    "global_fit_k_res",
    "global_fit_s_raw",
    "global_fit_s_used",
    "global_fit_sigma_s",
    "rescue_tail_q05",
    "rescue_tail_q50",
    "rescue_tail_q95",
    "rescue_tail_min",
    "rescue_tail_max",
    "rescue_tail_mean",
    "rescue_tail_std",
    "rescue_tail_contains_target_bin",
    "target_bin_tail_count",
    "target_bin_tail_fraction",
    "added_window",
    "added_center_x",
    "added_k",
    "used_steps",
]

_PMF_QUALITY_COLS = [
    "stage",
    "used_steps",
    "used_ksteps",
    "analysis_xmin",
    "analysis_xmax",
    "n_interest_bins",
    "n_covered_bins",
    "n_uncovered_bins",
    "first_uncovered_x",
    "last_uncovered_x",
    "uncovered_x_values",
    "coverage_fraction",
    "coverage_percent",
    "rmse_bestfit",
    "bestfit_offset",
    "n_error_bins",
    "mean_global_variance",
    "median_global_variance",
    "max_global_variance",
    "x_at_max_global_variance",
    "max_global_std",
]


def main() -> None:
    args = apply_quick_test_overrides(parse_args())
    if int(args.t_neq) <= 0:
        raise ValueError(
            "--t-neq must be > 0 because NEQ is required for offspring proposal, "
            "CFT/BAR stopping, and variance estimation."
        )
    system_root = Path(args.system_root).expanduser().resolve()
    bin_path = str(Path(args.bin).expanduser().resolve())
    run_context_path = system_root / "run_context.json"
    ctx = load_json(run_context_path)
    grid = build_grid(
        float(ctx["grid"]["xmin"]),
        float(ctx["grid"]["xmax"]),
        float(args.bin_width),
    )
    out_root = system_root / "MINES" / str(args.label) / "raw" / f"seed_{int(args.seed)}"
    out_root.mkdir(parents=True, exist_ok=True)
    generations_root = out_root / "generations"
    rescue_root = out_root / "rescue"
    windows_root = out_root / "windows"
    segment_store: dict[tuple[str, str], NEQSegment] = {}
    neq_patch_store: dict[str, PMFPatch] = {}
    neq_patch_status: dict[str, dict[str, Any]] = {}
    budget = BudgetTracker(total_budget_steps=int(args.total_budget_steps))

    generation_rows: list[dict[str, Any]] = []
    frontier_jsd_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    neq_quad_fit_rows: list[dict[str, Any]] = []
    skipped_segment_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    hs_fallback_rows: list[dict[str, Any]] = []
    growth_stop_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    fallback_global: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None

    def _flush_timing() -> None:
        try:
            write_timing_outputs(out_root, timing_rows)
        except Exception:
            pass

    atexit.register(_flush_timing)

    budget.spend(stage_cost_eq(int(args.n_eq_steps)), "L0", "EQ", "g00")
    with timed_operation(
        timing_rows,
        stage="g00",
        operation="run_eq_window",
        item="L0",
        metadata={"side": "left", "center_x": float(ctx["basins"]["left"]), "k": float(args.k_min), "n_eq_steps": int(args.n_eq_steps)},
    ):
        left0 = run_eq_window(
            name="L0",
            center_x=float(ctx["basins"]["left"]),
            k=float(args.k_min),
            generation=0,
            side="left",
            bin_path=bin_path,
            ctx=ctx,
            grid=grid,
            n_eq_steps=int(args.n_eq_steps),
            eq_save_every=int(args.eq_save_every),
            tail_fraction=float(args.tail_fraction),
            seed=int(args.seed + 10),
            root=windows_root / window_dirname("L0"),
            out_root=out_root,
        )
    budget.spend(stage_cost_eq(int(args.n_eq_steps)), "R0", "EQ", "g00")
    with timed_operation(
        timing_rows,
        stage="g00",
        operation="run_eq_window",
        item="R0",
        metadata={"side": "right", "center_x": float(ctx["basins"]["right"]), "k": float(args.k_min), "n_eq_steps": int(args.n_eq_steps)},
    ):
        right0 = run_eq_window(
            name="R0",
            center_x=float(ctx["basins"]["right"]),
            k=float(args.k_min),
            generation=0,
            side="right",
            bin_path=bin_path,
            ctx=ctx,
            grid=grid,
            n_eq_steps=int(args.n_eq_steps),
            eq_save_every=int(args.eq_save_every),
            tail_fraction=float(args.tail_fraction),
            seed=int(args.seed + 20),
            root=windows_root / window_dirname("R0"),
            out_root=out_root,
        )
    windows = [left0, right0]
    left_frontier = left0
    right_frontier = right0
    stop_reason = "max_generations"

    endpoint_xmin = float(min(left0.center_x, right0.center_x))
    endpoint_xmax = float(max(left0.center_x, right0.center_x))
    analysis_xmin = endpoint_xmin if args.analysis_xmin is None else float(args.analysis_xmin)
    analysis_xmax = endpoint_xmax if args.analysis_xmax is None else float(args.analysis_xmax)
    analysis_xmin = max(float(analysis_xmin), endpoint_xmin)
    analysis_xmax = min(float(analysis_xmax), endpoint_xmax)
    if analysis_xmin >= analysis_xmax:
        raise ValueError(
            f"Invalid analysis bounds after clamping to [x_L0, x_R0]: "
            f"analysis_xmin={analysis_xmin}, analysis_xmax={analysis_xmax}."
        )

    instruction_summary = {
        "label": str(args.label),
        "seed": int(args.seed),
        "system_root": str(system_root),
        "bin": bin_path,
        "parameters": json_ready(vars(args)),
        "budget_interpretation": "honest_dynamics_steps",
        "output_root": str(out_root),
        "endpoint_xmin": float(endpoint_xmin),
        "endpoint_xmax": float(endpoint_xmax),
        "analysis_xmin": float(analysis_xmin),
        "analysis_xmax": float(analysis_xmax),
        "analysis_bounds_rule": "clamped_to_initial_endpoint_centers",
        "pmf_method": str(args.pmf_method),
        "cft_ddf_threshold": float(args.cft_ddf_threshold),
        "t_neq_validation": "required_positive",
        "window_center_rule": "continuous_clipped_not_grid_snapped",
        "barrier_crossing_rule": "disabled_gt_slope_aware",
    }
    write_json(out_root / "run_request.json", instruction_summary)

    for generation_idx in range(int(args.max_generations)):
        stage_name = f"g{generation_idx:02d}"
        frontier_js = pair_jsd_metrics(
            eq_tail_samples(left_frontier),
            eq_tail_samples(right_frontier),
            grid,
        )
        pair_js = float(frontier_js["pair_jsd"])
        frontier_row = {
            "generation": int(generation_idx),
            "stage": stage_name,
            "left_frontier": left_frontier.name,
            "right_frontier": right_frontier.name,
            **frontier_js,
            "js_threshold": float(args.js_threshold),
            "decision": "",
            "reason": "",
        }
        frontier_row["frontiers_crossed_diagnostic"] = bool(
            left_frontier.center_x >= right_frontier.center_x
            or left_frontier.mean_x >= right_frontier.mean_x
        )
        frontier_row["left_frontier_mean_x"] = float(left_frontier.mean_x)
        frontier_row["right_frontier_mean_x"] = float(right_frontier.mean_x)
        frontier_row["left_frontier_x_most"] = float(left_frontier.x_most)
        frontier_row["right_frontier_x_most"] = float(right_frontier.x_most)
        frontier_row["crossing_coordinate_rule"] = "center_x_or_mean_x"
        frontier_row["frontiers_overlap_diagnostic"] = bool(
            math.isfinite(pair_js) and pair_js <= float(args.js_threshold)
        )
        eq_growth_cost = 2 * stage_cost_eq(int(args.n_eq_steps))
        if not budget.can_spend(eq_growth_cost):
            frontier_row["decision"] = "stop"
            frontier_row["reason"] = "budget_before_growth_eq"
            frontier_jsd_rows.append(frontier_row)
            stop_reason = "budget_before_growth_eq"
            break
        neq_growth_budget = choose_affordable_neq_count(
            requested_n_neq=int(args.n_neq_traj),
            t_neq=int(args.t_neq),
            budget=BudgetTracker(
                total_budget_steps=int(budget.total_budget_steps - eq_growth_cost),
                used_steps=int(budget.used_steps),
            ),
            allow_partial=bool(args.allow_partial_neq_budget),
            min_neq_traj=int(args.min_neq_traj),
        )
        if not bool(neq_growth_budget["can_run"]):
            frontier_row["decision"] = "stop"
            frontier_row["reason"] = "budget_before_growth_neq"
            frontier_jsd_rows.append(frontier_row)
            stop_reason = "budget_before_growth_neq"
            break
        frontier_jsd_rows.append(frontier_row)

        pair_key = (left_frontier.name, right_frontier.name)
        active_segment = segment_store.get(pair_key)
        if active_segment is None:
            segment_name = f"SEG_{left_frontier.name}__{right_frontier.name}"
            budget.spend(
                int(neq_growth_budget["cost_actual"]),
                segment_name,
                "NEQ_PARTIAL" if bool(neq_growth_budget["neq_budget_limited"]) else "NEQ",
                stage_name,
            )
            with timed_operation(
                timing_rows,
                stage=stage_name,
                operation="run_neq_protocol",
                item=segment_name,
                metadata={
                    "n_neq_traj": int(neq_growth_budget["n_neq_traj_actual"]),
                    "t_neq": int(args.t_neq),
                    "stage_index": generation_idx,
                },
            ):
                active_segment = run_neq_protocol(
                    name=segment_name,
                    left=left_frontier,
                    right=right_frontier,
                    boundary_left=left_frontier,
                    boundary_right=right_frontier,
                    bin_path=bin_path,
                    ctx=ctx,
                    t_neq=int(args.t_neq),
                    n_neq_traj=int(neq_growth_budget["n_neq_traj_actual"]),
                    seed=int(args.seed + 10000 + generation_idx),
                    root=out_root / "segments" / segment_name,
                    k_min=float(args.k_min),
                    k_max=float(args.k_max),
                    out_root=out_root,
                    neq_protocol_mode=str(args.neq_protocol_mode),
                    n_neq_traj_requested=int(neq_growth_budget["n_neq_traj_requested"]),
                    neq_budget_limited=bool(neq_growth_budget["neq_budget_limited"]),
                    neq_cost_requested=int(neq_growth_budget["cost_requested"]),
                    neq_cost_actual=int(neq_growth_budget["cost_actual"]),
                    remaining_budget_before_segment=int(neq_growth_budget["remaining_budget_before_segment"]),
                )
            segment_store[pair_key] = active_segment
        ensure_segment_connectivity(
            active_segment,
            grid,
            float(args.neq_connectivity_threshold),
            out_root,
            mts_patch_built=False,
            reason="growth_segment_connectivity_checked",
        )

        cft_now = compute_segment_cft_summary(active_segment, ctx)
        cft_delta_f_now = cft_now.get("cft_delta_f")
        # CFT kept as diagnostic only — not the stop trigger
        stop_by_cft = False
        # Growth stop: quantile crossing of EoP or EQ tail distributions
        eop_cross = neq_eop_quantile_crossing(active_segment)
        eq_cross = eq_tail_quantile_crossing(left_frontier, right_frontier)
        stop_by_eop = bool(eop_cross["crossed"])
        stop_by_eq = bool(eq_cross["crossed"])
        stop_by_quantile = stop_by_eop or stop_by_eq
        if stop_by_eop and stop_by_eq:
            _stop_reason_quantile = "eop_and_eq_quantile_crossing"
        elif stop_by_eop:
            _stop_reason_quantile = "eop_quantile_crossing"
        elif stop_by_eq:
            _stop_reason_quantile = "eq_quantile_crossing"
        else:
            _stop_reason_quantile = ""
        growth_stop_row = {
            "generation": int(generation_idx),
            "stage": stage_name,
            "active_segment": active_segment.name,
            **cft_now,
            "cft_ddf_threshold": float(args.cft_ddf_threshold),
            "stop_by_cft": False,
            "stop_by_eop_crossing": bool(stop_by_eop),
            "stop_by_eq_crossing": bool(stop_by_eq),
            "forward_eop_q95": float(eop_cross["fwd_q_high"]),
            "reverse_eop_q05": float(eop_cross["rev_q_low"]),
            "left_eq_q95": float(eq_cross["left_q_high"]),
            "right_eq_q05": float(eq_cross["right_q_low"]),
            "crossing_quantile_low": float(eop_cross["q_low"]),
            "crossing_quantile_high": float(eop_cross["q_high"]),
            "growth_stop_reason": _stop_reason_quantile,
            "stop_reason": _stop_reason_quantile,
            "used_steps": int(budget.used_steps),
        }
        growth_stop_rows.append(growth_stop_row)
        if stop_by_quantile:
            frontier_row["decision"] = "stop"
            frontier_row["reason"] = _stop_reason_quantile
            stop_reason = _stop_reason_quantile
            break
        frontier_row["decision"] = "grow"
        frontier_row["reason"] = (
            "continue_growth_partial_neq"
            if bool(neq_growth_budget["neq_budget_limited"])
            else "continue_growth"
        )

        left_endpoints = endpoint_x_from_trajectories(active_segment.forward_trajectories)
        right_endpoints = endpoint_x_from_trajectories(active_segment.reverse_trajectories)
        left_center_x, left_k, left_plan = propose_child_from_neq_endpoints(
            side="left",
            parent=left_frontier,
            opposite=right_frontier,
            endpoint_x=left_endpoints,
            grid=grid,
            q_next=float(args.q_next),
            alpha=float(args.alpha),
            x_leap=float(args.x_leap),
            k_min=float(args.k_min),
            k_max=float(args.k_max),
        )
        right_center_x, right_k, right_plan = propose_child_from_neq_endpoints(
            side="right",
            parent=right_frontier,
            opposite=left_frontier,
            endpoint_x=right_endpoints,
            grid=grid,
            q_next=float(args.q_next),
            alpha=float(args.alpha),
            x_leap=float(args.x_leap),
            k_min=float(args.k_min),
            k_max=float(args.k_max),
        )
        left_plan["name"] = f"L{generation_idx + 1}"
        right_plan["name"] = f"R{generation_idx + 1}"
        generation_dir = generations_root / stage_name
        write_generation_side_outputs(
            generation_dir / "left",
            active_segment.forward_path_file,
            active_segment.forward_trajectory_files,
            active_segment.forward_trajectories,
            out_root,
            left_plan,
        )
        write_generation_side_outputs(
            generation_dir / "right",
            active_segment.reverse_path_file,
            active_segment.reverse_trajectory_files,
            active_segment.reverse_trajectories,
            out_root,
            right_plan,
        )
        budget.spend(stage_cost_eq(int(args.n_eq_steps)), left_plan["name"], "EQ", stage_name)
        with timed_operation(
            timing_rows,
            stage=stage_name,
            operation="run_eq_window",
            item=left_plan["name"],
            metadata={"side": "left", "center_x": float(left_center_x), "k": float(left_k), "n_eq_steps": int(args.n_eq_steps)},
        ):
            left_child = run_eq_window(
                name=left_plan["name"],
                center_x=float(left_center_x),
                k=float(left_k),
                generation=generation_idx + 1,
                side="left",
                bin_path=bin_path,
                ctx=ctx,
                grid=grid,
                n_eq_steps=int(args.n_eq_steps),
                eq_save_every=int(args.eq_save_every),
                tail_fraction=float(args.tail_fraction),
                seed=int(args.seed + 20000 + generation_idx),
                root=windows_root / window_dirname(left_plan["name"]),
                out_root=out_root,
            )
        budget.spend(stage_cost_eq(int(args.n_eq_steps)), right_plan["name"], "EQ", stage_name)
        with timed_operation(
            timing_rows,
            stage=stage_name,
            operation="run_eq_window",
            item=right_plan["name"],
            metadata={"side": "right", "center_x": float(right_center_x), "k": float(right_k), "n_eq_steps": int(args.n_eq_steps)},
        ):
            right_child = run_eq_window(
                name=right_plan["name"],
                center_x=float(right_center_x),
                k=float(right_k),
                generation=generation_idx + 1,
                side="right",
                bin_path=bin_path,
                ctx=ctx,
                grid=grid,
                n_eq_steps=int(args.n_eq_steps),
                eq_save_every=int(args.eq_save_every),
                tail_fraction=float(args.tail_fraction),
                seed=int(args.seed + 30000 + generation_idx),
                root=windows_root / window_dirname(right_plan["name"]),
                out_root=out_root,
            )
        windows.extend([left_child, right_child])
        generation_row = {
            "generation": int(generation_idx),
            "stage": stage_name,
            "left_parent": left_frontier.name,
            "right_parent": right_frontier.name,
            "active_segment": active_segment.name,
            **frontier_js,
            "n_neq_traj_requested": int(active_segment.n_neq_traj_requested),
            "n_neq_traj_actual": int(active_segment.n_neq_traj_actual),
            "neq_budget_limited": int(bool(active_segment.neq_budget_limited)),
            "left_child": left_child.name,
            "right_child": right_child.name,
            "left_target_x": left_plan["target_x"],
            "right_target_x": right_plan["target_x"],
            "left_target_source": left_plan["target_source"],
            "right_target_source": right_plan["target_source"],
            "left_barrier_crossing": left_plan["barrier_crossing_diagnostic"],
            "right_barrier_crossing": right_plan["barrier_crossing_diagnostic"],
            "left_k_rule": left_plan["k_rule"],
            "right_k_rule": right_plan["k_rule"],
            "left_child_k": float(left_child.k),
            "right_child_k": float(right_child.k),
            "left_barrier_crossing_displacement": left_plan["barrier_crossing_displacement"],
            "right_barrier_crossing_displacement": right_plan["barrier_crossing_displacement"],
            "left_barrier_crossing_tol": left_plan["barrier_crossing_tol"],
            "right_barrier_crossing_tol": right_plan["barrier_crossing_tol"],
            "left_center_raw": left_plan.get("center_raw", ""),
            "left_center_x": float(left_child.center_x),
            "right_center_raw": right_plan.get("center_raw", ""),
            "right_center_x": float(right_child.center_x),
            "pmf_method": str(args.pmf_method),
            "cft_solved_once": bool(cft_now["cft_solved_once"]),
            "cft_delta_f": cft_now.get("cft_delta_f"),
            "cft_delta_f_unc": cft_now.get("cft_delta_f_unc"),
            "cft_method": cft_now.get("cft_method", ""),
            "cft_reason": cft_now.get("cft_reason", ""),
            "cft_ddf_threshold": float(args.cft_ddf_threshold),
            "stop_by_cft": False,
            "stop_by_eop_crossing": bool(stop_by_eop),
            "stop_by_eq_crossing": bool(stop_by_eq),
            "growth_stop_reason": _stop_reason_quantile,
            "used_steps": int(budget.used_steps),
        }
        generation_rows.append(generation_row)
        write_json(
            generation_dir / "generation_summary.json",
            {
                **generation_row,
                "left_child_design_file": relative_to_root(generation_dir / "left" / "child_design.json", out_root),
                "right_child_design_file": relative_to_root(generation_dir / "right" / "child_design.json", out_root),
            },
        )
        left_frontier = left_child
        right_frontier = right_child
    else:
        stop_reason = "max_generations"

    with timed_operation(
        timing_rows,
        stage="reconstruct_growth",
        operation="reconstruct_chain",
        item="all_windows",
        metadata={"n_windows": len(windows), "n_segments_before": len(segment_store)},
    ):
        clusters, segments, patches, global_pmf, global_variance, fit_details, js_rows, patches_for_global = reconstruct_chain(
            windows=windows,
            segment_store=segment_store,
            neq_patch_store=neq_patch_store,
            neq_patch_status=neq_patch_status,
            args=args,
            ctx=ctx,
            grid=grid,
            out_root=out_root,
            bin_path=bin_path,
            base_seed=int(args.seed),
            stage_index=0,
            budget=budget,
            stage_label="reconstruct_growth",
            skipped_segment_rows=skipped_segment_rows,
            timing_rows=timing_rows,
        )
    all_segments = sorted(segment_store.values(), key=lambda item: item.name)
    quality_rows.append(
        compute_pmf_quality_metrics(
            grid=grid,
            global_pmf=global_pmf,
            global_variance=global_variance,
            ctx=ctx,
            used_steps=int(budget.used_steps),
            stage="growth_reconstruct",
            analysis_xmin=float(analysis_xmin),
            analysis_xmax=float(analysis_xmax),
        )
    )

    write_csv(
        out_root / "generation_summary.csv",
        ordered_fieldnames(
            generation_rows,
            extras=[
                "generation", "stage", "pair_jsd_raw", "pair_jsd_norm", "pair_jsd",
                "pmf_method", "cft_solved_once", "cft_delta_f", "cft_delta_f_unc",
                "cft_method", "cft_reason", "cft_ddf_threshold", "stop_by_cft",
                "left_center_raw", "left_center_x", "right_center_raw", "right_center_x",
                "left_k_rule", "right_k_rule", "used_steps",
            ],
        ),
        generation_rows,
    )
    write_csv(
        out_root / "growth_stop_summary.csv",
        ordered_fieldnames(
            growth_stop_rows,
            extras=[
                "generation", "stage", "active_segment",
                "cft_solved_once", "cft_delta_f", "cft_delta_f_unc",
                "cft_method", "cft_reason", "cft_ddf_threshold",
                "stop_by_cft", "stop_reason", "used_steps",
            ],
        ),
        growth_stop_rows,
    )
    write_csv(
        out_root / "frontier_jsd.csv",
        ordered_fieldnames(
            frontier_jsd_rows,
            extras=[
                "generation",
                "stage",
                "left_frontier",
                "right_frontier",
                "pair_jsd_raw",
                "pair_jsd_norm",
                "pair_jsd",
                "js_threshold",
                "decision",
                "reason",
            ],
        ),
        frontier_jsd_rows,
    )
    write_state_tables(
        out_root,
        out_root,
        windows,
        clusters,
        all_segments,
        patches,
        fit_details,
        js_rows,
        grid,
        global_pmf,
        global_variance,
        ctx,
        hs_fallback_rows=hs_fallback_rows,
        fallback_global=fallback_global,
    )
    write_csv(
        out_root / "skipped_segments.csv",
        ordered_fieldnames(
            skipped_segment_rows,
            extras=[
                "segment",
                "left_boundary",
                "right_boundary",
                "reason",
                "n_neq_traj_requested",
                "n_neq_traj_affordable",
                "min_neq_traj",
                "cost_requested",
                "remaining_budget_before_segment",
                "used_steps",
                "budget_steps",
            ],
        ),
        skipped_segment_rows,
    )
    write_csv(
        out_root / "pmf_quality_vs_steps.csv",
        ordered_fieldnames(quality_rows, extras=_PMF_QUALITY_COLS),
        quality_rows,
    )
    budget.write(out_root / "budget_ledger.csv")

    rescue_counter = 0
    while args.max_rescue_rounds is None or rescue_counter < int(args.max_rescue_rounds):
        target_info: dict[str, Any] | None = None
        with timed_operation(
            timing_rows,
            stage=f"rescue_round_{rescue_counter + 1:02d}_pre",
            operation="choose_rescue_target_priority",
            item="rescue_target",
            metadata={"n_windows": len(windows), "n_segments": len(all_segments)},
        ):
            target_info = choose_rescue_target_priority(
                grid=grid,
                global_pmf=global_pmf,
                global_variance=global_variance,
                analysis_xmin=float(analysis_xmin),
                analysis_xmax=float(analysis_xmax),
                skipped_segment_rows=skipped_segment_rows,
                mts_failed_segments=mts_failed_segments(all_segments),
            )
        if target_info is None:
            stop_reason = "no_rescue_target_available"
            break
        rescue_counter += 1
        rescue_stage = f"rescue_round_{rescue_counter:02d}"
        target_x = float(target_info["target_x"])
        target_variance = float(target_info.get("target_variance", float("nan")))
        target_priority = str(target_info.get("target_priority", "finite_variance"))
        target_reason = str(target_info.get("target_reason", "max_finite_global_variance"))
        uncovered_start_x = float(target_info.get("uncovered_start_x", float("nan")))
        uncovered_end_x = float(target_info.get("uncovered_end_x", float("nan")))
        uncovered_width = float(target_info.get("uncovered_width", float("nan")))
        uncovered_n_bins = int(target_info.get("uncovered_n_bins", 0))
        if not budget.can_spend(stage_cost_eq(int(args.n_eq_steps))):
            stop_reason = "budget_before_rescue"
            break
        rescue_design: dict[str, Any] = {}
        _rescue_round_root = rescue_root / f"rescue_round_{rescue_counter:02d}"
        _n_fit_rows_before = len(neq_quad_fit_rows)
        with timed_operation(
            timing_rows,
            stage=rescue_stage,
            operation="design_rescue_window_mean_only_gt",
            item="rescue_design",
            metadata={"target_x": target_x, "target_priority": target_priority},
        ):
            rescue_design = design_rescue_window_mean_only_gt(
                target_info=target_info,
                windows=windows,
                all_segments=all_segments,
                neq_patch_store=neq_patch_store,
                neq_quad_fit_rows=neq_quad_fit_rows,
                global_pmf=global_pmf,
                global_variance=global_variance,
                grid=grid,
                args=args,
                analysis_xmin=float(analysis_xmin),
                analysis_xmax=float(analysis_xmax),
                rescue_rows=rescue_rows,
                ctx=ctx,
                rescue_round_root=_rescue_round_root,
            )
        for _fit_row in neq_quad_fit_rows[_n_fit_rows_before:]:
            _fit_row.setdefault("round", int(rescue_counter))
        rescue_center_x = float(rescue_design["rescue_center_x"])
        rescue_k = float(rescue_design["rescue_k"])
        rescue_name = f"M{rescue_counter}"
        budget.spend(stage_cost_eq(int(args.n_eq_steps)), rescue_name, "EQ", rescue_stage)
        with timed_operation(
            timing_rows,
            stage=rescue_stage,
            operation="run_eq_window",
            item=rescue_name,
            metadata={
                "side": "rescue",
                "center_x": rescue_center_x,
                "k": rescue_k,
                "n_eq_steps": int(args.n_eq_steps),
            },
        ):
            rescue_window = run_eq_window(
                name=rescue_name,
                center_x=rescue_center_x,
                k=rescue_k,
                generation=len(generation_rows) + rescue_counter,
                side="rescue",
                bin_path=bin_path,
                ctx=ctx,
                grid=grid,
                n_eq_steps=int(args.n_eq_steps),
                eq_save_every=int(args.eq_save_every),
                tail_fraction=float(args.tail_fraction),
                seed=int(args.seed + 50000 + rescue_counter),
                root=windows_root / window_dirname(rescue_name),
                out_root=out_root,
            )
        windows.append(rescue_window)
        _rescue_grid_dx = (
            abs(float(np.asarray(grid, dtype=float)[1] - np.asarray(grid, dtype=float)[0]))
            if len(grid) > 1
            else float(getattr(args, "bin_width", 0.1))
        )
        _rescue_target_bin_x = float(rescue_design["target_bin_x"])
        _x_tail = eq_tail_samples(rescue_window)
        if len(_x_tail) > 0:
            _in_target = np.abs(_x_tail - _rescue_target_bin_x) <= 0.5 * _rescue_grid_dx
            _tbc = int(np.sum(_in_target))
            _tail_stats: dict[str, Any] = {
                "rescue_tail_q05": float(np.quantile(_x_tail, 0.05)),
                "rescue_tail_q50": float(np.quantile(_x_tail, 0.50)),
                "rescue_tail_q95": float(np.quantile(_x_tail, 0.95)),
                "rescue_tail_min": float(np.min(_x_tail)),
                "rescue_tail_max": float(np.max(_x_tail)),
                "rescue_tail_mean": float(np.mean(_x_tail)),
                "rescue_tail_std": float(np.std(_x_tail)),
                "rescue_tail_contains_target_bin": bool(_tbc > 0),
                "target_bin_tail_count": _tbc,
                "target_bin_tail_fraction": float(_tbc) / float(len(_x_tail)),
            }
        else:
            _tail_stats = {
                "rescue_tail_q05": float("nan"),
                "rescue_tail_q50": float("nan"),
                "rescue_tail_q95": float("nan"),
                "rescue_tail_min": float("nan"),
                "rescue_tail_max": float("nan"),
                "rescue_tail_mean": float("nan"),
                "rescue_tail_std": float("nan"),
                "rescue_tail_contains_target_bin": False,
                "target_bin_tail_count": 0,
                "target_bin_tail_fraction": float("nan"),
            }
        with timed_operation(
            timing_rows,
            stage=rescue_stage,
            operation="reconstruct_chain",
            item="all_windows",
            metadata={"n_windows": len(windows), "n_segments_before": len(segment_store)},
        ):
            clusters, segments, patches, global_pmf, global_variance, fit_details, js_rows, patches_for_global = reconstruct_chain(
                windows=windows,
                segment_store=segment_store,
                neq_patch_store=neq_patch_store,
                neq_patch_status=neq_patch_status,
                args=args,
                ctx=ctx,
                grid=grid,
                out_root=out_root,
                bin_path=bin_path,
                base_seed=int(args.seed),
                stage_index=rescue_counter,
                budget=budget,
                stage_label=rescue_stage,
                skipped_segment_rows=skipped_segment_rows,
                timing_rows=timing_rows,
            )
        all_segments = sorted(segment_store.values(), key=lambda item: item.name)
        quality_rows.append(
            compute_pmf_quality_metrics(
                grid=grid,
                global_pmf=global_pmf,
                global_variance=global_variance,
                ctx=ctx,
                used_steps=int(budget.used_steps),
                stage=rescue_stage,
                analysis_xmin=float(analysis_xmin),
                analysis_xmax=float(analysis_xmax),
            )
        )
        rescue_row: dict[str, Any] = {
            "round": int(rescue_counter),
            "target_priority": target_priority,
            "target_reason": target_reason,
            "x_rescue_target": rescue_design["x_rescue_target"],
            "target_bin_x": rescue_design["target_bin_x"],
            "target_variance": float(target_variance),
            "uncovered_start_x": float(uncovered_start_x),
            "uncovered_end_x": float(uncovered_end_x),
            "uncovered_width": float(uncovered_width),
            "uncovered_n_bins": int(uncovered_n_bins),
            "rescue_center_x_raw": rescue_design["rescue_center_x_raw"],
            "rescue_center_x": rescue_design["rescue_center_x"],
            "rescue_center_clamped_to_bounds": rescue_design["rescue_center_clamped_to_bounds"],
            "sigma_target": rescue_design["sigma_target"],
            "k_from_sigma": rescue_design["k_from_sigma"],
            "rescue_k_base": rescue_design["rescue_k_base"],
            "rescue_k": rescue_design["rescue_k"],
            "rescue_k_unclipped": rescue_design["rescue_k_unclipped"],
            "rescue_k_saturated": rescue_design["rescue_k_saturated"],
            "rescue_k_rule": rescue_design["rescue_k_rule"],
            "rescue_center_rule": rescue_design["rescue_center_rule"],
            "matched_child_name": rescue_design["matched_child_name"],
            "matched_child_side": rescue_design["matched_child_side"],
            "matched_child_center_x": rescue_design["matched_child_center_x"],
            "matched_child_target_x": rescue_design.get("matched_child_target_x", rescue_design.get("matched_target_x", float("nan"))),
            "matched_target_distance": rescue_design["matched_target_distance"],
            "matched_child_k": rescue_design["matched_child_k"],
            "matched_child_raw_k": rescue_design["matched_child_raw_k"],
            "matched_child_k_rule": rescue_design["matched_child_k_rule"],
            "matched_child_target_source": rescue_design["matched_child_target_source"],
            "matched_child_used_for_center": rescue_design["matched_child_used_for_center"],
            "rescue_retry_count": rescue_design["rescue_retry_count"],
            "rescue_scale": rescue_design["rescue_scale"],
            "rescue_k_retry_rule": rescue_design.get("rescue_k_retry_rule", ""),
            **{k: v for k, v in rescue_design.items() if k.startswith("gt_")},
            **{k: v for k, v in rescue_design.items() if k.startswith("mo_")},
            **{k: v for k, v in rescue_design.items() if k.startswith("rescue_fit_") or k.startswith("rescue_gt_") or k == "rescue_background_fit_method" or k == "rescue_design_rule"},
            **{k: v for k, v in rescue_design.items() if k.startswith("global_fit_")},
            **_tail_stats,
            "added_window": rescue_name,
            "added_center_x": float(rescue_window.center_x),
            "added_k": float(rescue_window.k),
            "used_steps": int(budget.used_steps),
        }
        rescue_rows.append(rescue_row)
        write_csv(
            out_root / "rescue_summary.csv",
            ordered_fieldnames(
                rescue_rows,
                extras=_RESCUE_SUMMARY_COLS,
            ),
            rescue_rows,
        )
        if neq_quad_fit_rows:
            write_csv(
                out_root / "rescue_neq_quadratic_fits.csv",
                ordered_fieldnames(
                    neq_quad_fit_rows,
                    extras=[
                        "round", "segment", "patch_name", "patch_kind", "fit_source",
                        "fit_accepted", "fallback_reason", "n_fit_bins",
                        "x_fit_min", "x_fit_max", "k0", "x0", "F0",
                        "a", "b", "c", "weighted_rmse", "reduced_chi2", "variance_floor",
                    ],
                ),
                neq_quad_fit_rows,
            )
            _rescue_round_root.mkdir(parents=True, exist_ok=True)
            write_json(
                _rescue_round_root / "neq_quadratic_fit_summary.json",
                [
                    {k: json_ready(v) for k, v in row.items()}
                    for row in neq_quad_fit_rows
                    if row.get("round") == int(rescue_counter)
                ],
            )
        with timed_operation(
            timing_rows,
            stage=rescue_stage,
            operation="write_state_tables",
            item="rescue",
            metadata={"n_windows": len(windows), "n_patches": len(patches)},
        ):
            write_state_tables(
                out_root,
                out_root,
                windows,
                clusters,
                all_segments,
                patches,
                fit_details,
                js_rows,
                grid,
                global_pmf,
                global_variance,
                ctx,
                hs_fallback_rows=hs_fallback_rows,
                fallback_global=fallback_global,
            )
        write_csv(
            out_root / "skipped_segments.csv",
            ordered_fieldnames(
                skipped_segment_rows,
                extras=[
                    "segment",
                    "left_boundary",
                    "right_boundary",
                    "reason",
                    "n_neq_traj_requested",
                    "n_neq_traj_affordable",
                    "min_neq_traj",
                    "cost_requested",
                    "remaining_budget_before_segment",
                    "used_steps",
                    "budget_steps",
                ],
            ),
            skipped_segment_rows,
        )
        write_csv(
            out_root / "pmf_quality_vs_steps.csv",
            ordered_fieldnames(quality_rows, extras=_PMF_QUALITY_COLS),
            quality_rows,
        )
        budget.write(out_root / "budget_ledger.csv")
        write_timing_outputs(out_root, timing_rows)
        rescue_round_root = rescue_root / f"rescue_round_{rescue_counter:02d}"
        rescue_round_root.mkdir(parents=True, exist_ok=True)
        write_json(
            rescue_round_root / "rescue_decision.json",
            {**target_info, **rescue_design,
             "added_window": rescue_name,
             "added_center_x": float(rescue_window.center_x),
             "added_k": float(rescue_window.k),
             "used_steps": int(budget.used_steps)},
        )
        write_state_snapshot(
            rescue_round_root,
            out_root,
            windows,
            clusters,
            segments,
            patches,
            fit_details,
            js_rows,
            grid,
            global_pmf,
            global_variance,
            ctx,
            hs_fallback_rows=hs_fallback_rows,
            fallback_global=fallback_global,
            neq_patch_store=neq_patch_store,
            neq_patch_status=neq_patch_status,
            patches_for_global=patches_for_global,
        )
    else:
        stop_reason = "max_rescue_rounds"

    if not rescue_rows:
        write_csv(out_root / "rescue_summary.csv", _RESCUE_SUMMARY_COLS, [])

    if not neq_quad_fit_rows:
        write_csv(
            out_root / "rescue_neq_quadratic_fits.csv",
            [
                "round", "segment", "patch_name", "patch_kind", "fit_source",
                "fit_accepted", "fallback_reason", "n_fit_bins",
                "x_fit_min", "x_fit_max", "k0", "x0", "F0",
                "a", "b", "c", "weighted_rmse", "reduced_chi2", "variance_floor",
            ],
            [],
        )

    (
        clusters,
        patches_for_global,
        global_pmf,
        global_variance,
        fit_details,
        js_rows,
        _,
        _ext_rows,
    ) = run_final_eq_extension_refinement(
        windows=windows,
        clusters=clusters,
        segment_store=segment_store,
        neq_patch_store=neq_patch_store,
        neq_patch_status=neq_patch_status,
        patches_for_global=patches_for_global,
        skipped_segment_rows=skipped_segment_rows,
        args=args,
        ctx=ctx,
        grid=grid,
        out_root=out_root,
        bin_path=bin_path,
        budget=budget,
        timing_rows=timing_rows,
        quality_rows=quality_rows,
        global_pmf=global_pmf,
        global_variance=global_variance,
        fit_details=fit_details,
        js_rows=js_rows,
        analysis_xmin=float(analysis_xmin),
        analysis_xmax=float(analysis_xmax),
    )

    _last_ext_reason = _ext_rows[-1].get("stop_reason", "") if _ext_rows else ""
    if _last_ext_reason == "eq_connectivity_lost":
        fallback_after_eq_connectivity_lost(clusters=clusters, windows=windows, out_root=out_root)

    budget.write(out_root / "budget_ledger.csv")

    all_segments_sorted = sorted(segment_store.values(), key=lambda item: item.name)
    partial_neq_segments = [
        {
            "segment": segment.name,
            "n_neq_traj_requested": int(segment.n_neq_traj_requested),
            "n_neq_traj_actual": int(segment.n_neq_traj_actual),
        }
        for segment in all_segments_sorted
        if bool(segment.neq_budget_limited)
    ]
    eligible_neq_segments = [
        segment
        for segment in all_segments_sorted
        if segment.neq_patch_decision or segment.mts_patch_built or segment.cft_summary
    ]
    disconnected = disconnected_segments(all_segments_sorted)
    mts_failed = mts_failed_segments(eligible_neq_segments)
    if mts_failed and budget_exhausted_for_future_sampling(args, budget):
        with timed_operation(
            timing_rows,
            stage="post_growth",
            operation="build_hs_fallback_patches",
            item="mts_failed_segments",
            metadata={"n_mts_failed": int(len(mts_failed))},
        ):
            hs_patches, hs_fallback_rows = build_hs_fallback_patches(
                disconnected_segments=mts_failed,
                grid=grid,
                ctx=ctx,
                n_boot=int(args.n_bootstrap_neq),
                base_seed=int(args.seed),
                out_root=out_root,
            )
        if hs_patches:
            fallback_patches = list(patches) + hs_patches
            fallback_global_pmf, fallback_global_variance, fallback_fit_details = fit_global_pmf_from_patches(
                fallback_patches,
                grid,
                float(args.variance_floor),
                reference_x=None,
            )
            write_patch_offset_aligned_outputs(hs_patches, fallback_fit_details)
            fallback_global = (fallback_global_pmf, fallback_global_variance, fallback_fit_details)
            quality_rows.append(
                compute_pmf_quality_metrics(
                    grid=grid,
                    global_pmf=fallback_global_pmf,
                    global_variance=fallback_global_variance,
                    ctx=ctx,
                    used_steps=int(budget.used_steps),
                    stage="hs_fallback",
                    analysis_xmin=float(analysis_xmin),
                    analysis_xmax=float(analysis_xmax),
                )
            )
        write_state_tables(
            out_root,
            out_root,
            windows,
            clusters,
            all_segments_sorted,
            patches,
            fit_details,
            js_rows,
            grid,
            global_pmf,
            global_variance,
            ctx,
            hs_fallback_rows=hs_fallback_rows,
            fallback_global=fallback_global,
        )
        write_csv(
            out_root / "pmf_quality_vs_steps.csv",
            ordered_fieldnames(quality_rows, extras=_PMF_QUALITY_COLS),
            quality_rows,
        )

    write_json(
        out_root / "mines_variance_fusion_summary.json",
        {
            "label": str(args.label),
            "seed": int(args.seed),
            "stop_reason": stop_reason,
            "used_steps": int(budget.used_steps),
            "budget_steps": int(args.total_budget_steps),
            "n_windows": int(len(windows)),
            "n_clusters": int(len(clusters)),
            "n_segments": int(len(all_segments_sorted)),
            "n_patches": int(len(patches)),
            "n_disconnected_segments": int(len(disconnected)),
            "n_neq_segments_total": int(len(eligible_neq_segments)),
            "n_neq_mts_patches_built": int(sum(1 for segment in eligible_neq_segments if bool(segment.mts_patch_built))),
            "n_neq_segments_eop_crossed": int(sum(1 for segment in eligible_neq_segments if bool(segment.connectivity.get("eop_crossed", False)))),
            "n_neq_segments_eop_not_crossed_but_mts_built": int(sum(1 for segment in eligible_neq_segments if (not bool(segment.connectivity.get("eop_crossed", False))) and bool(segment.mts_patch_built))),
            "n_neq_segments_cft_solved_once": int(sum(1 for segment in eligible_neq_segments if bool(segment.cft_summary.get("cft_solved_once", False)))),
            "n_neq_segments_bootstrap_recomputed_cft": int(sum(1 for segment in eligible_neq_segments if bool(segment.cft_summary.get("bootstrap_recomputed_cft", False)))),
            "n_mts_failed_segments": int(len(mts_failed)),
            "n_hs_fallback_patches": int(len(hs_fallback_rows)),
            "allow_partial_neq_budget": bool(args.allow_partial_neq_budget),
            "min_neq_traj": int(args.min_neq_traj),
            "n_partial_neq_segments": int(len(partial_neq_segments)),
            "n_skipped_segments": int(len(skipped_segment_rows)),
            "pmf_quality_vs_steps": "pmf_quality_vs_steps.csv",
            "endpoint_xmin": float(endpoint_xmin),
            "endpoint_xmax": float(endpoint_xmax),
            "analysis_xmin": float(analysis_xmin),
            "analysis_xmax": float(analysis_xmax),
            "analysis_bounds_rule": "clamped_to_initial_endpoint_centers",
            "partial_neq_segments": partial_neq_segments,
            "generation_count": int(len(generation_rows)),
            "rescue_round_count": int(len(rescue_rows)),
            "pmf_method": str(args.pmf_method),
            "cft_ddf_threshold": float(args.cft_ddf_threshold),
            "growth_stop_rule": "cft_delta_f_below_threshold",
            "window_center_rule": "continuous_clipped_not_grid_snapped",
            "barrier_crossing_rule": "disabled_gt_slope_aware",
            "rescue_strategy": "choose_rescue_target_priority",
            "rescue_target_rule": "uncovered_interval_first_then_failed_gap_then_max_finite_variance",
            "connectivity_refinement_strategy": "adaptive_connectivity_refinement",
            "neq_gt_strategy": "endpoint_local_harmonic_interpolation",
            "intercluster_boundary_rule": "existing_connected_segment_to_right_left_boundary_most_left_else_nearest_boundary",
            "coverage_rule": "actual_patch_data_support",
            "neq_protocol_mode": str(args.neq_protocol_mode).upper(),
            "summary_files": {
                "windows": "windows.csv",
                "clusters": "clusters.csv",
                "segments": "segments.csv",
                "disconnected_segments": "disconnected_segments.csv",
                "neq_patch_decisions": "neq_patch_decisions.csv",
                "mts_failed_segments": "mts_failed_segments.csv",
                "patches": "patches.csv",
                "all_neq_patches": "all_neq_patches.csv",
                "patches_used_for_global_fit": "patches_used_for_global_fit.csv",
                "generation_summary": "generation_summary.csv",
                "growth_stop_summary": "growth_stop_summary.csv",
                "frontier_jsd": "frontier_jsd.csv",
                "rescue_summary": "rescue_summary.csv",
                "skipped_segments": "skipped_segments.csv",
                "budget_ledger": "budget_ledger.csv",
                "pmf_quality_vs_steps": "pmf_quality_vs_steps.csv",
                "global_pmf": "global_pmf.csv",
                "global_fit_summary": "global_fit_summary.json",
                "hs_fallback_segments": "hs_fallback_segments.csv",
                "operation_timing": "operation_timing.csv",
                "operation_timing_summary": "operation_timing_summary.csv",
                "global_pmf_with_hs_fallback": (
                    "global_pmf_with_hs_fallback.csv" if fallback_global is not None else None
                ),
                "global_fit_summary_with_hs_fallback": (
                    "global_fit_summary_with_hs_fallback.json" if fallback_global is not None else None
                ),
            },
        },
    )
    write_timing_outputs(out_root, timing_rows)
    print(str(out_root / "mines_variance_fusion_summary.json"))


if __name__ == "__main__":
    main()
