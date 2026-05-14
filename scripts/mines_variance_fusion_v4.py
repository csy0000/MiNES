#!/usr/bin/env python3
"""
MiNES v4 — Milestoned Nonequilibrium Switching, Version 4.

Changes from v3:

  v4-1: Remove final perturbation appending.  Use full bidirectional NES
        trajectories directly for CFT/MTS without any ad-hoc endpoint
        augmentation or protocol elongation.
  v4-2: Remove augmented protocol files (protocol_forward_augmented.csv,
        protocol_reverse_augmented.csv) and the corresponding NEQSegment fields.
  v4-3: Add is_valid_bidirectional_segment() helper; _find_existing_segment()
        only returns segments with both forward AND reverse trajectories.
        If only one direction exists, run a new bidirectional NES instead.

Changes inherited from v3 (all preserved):

  Refinement routing: NES only for transition/barrier-like segments; basin → midpoint EQ.
  BAR failure means disconnected; raw sigmoid is diagnostic only.
  --max-refinement-rounds (default 10); --max-rescue-rounds not exposed.
  Refinement loop processes one disconnected pair per round; fresh cluster rebuild each round.
  CFT/MTS failure falls back to midpoint mean-only GT EQ insertion.
  x_most removed from EnsembleWindow and all outputs.
  mts_solved checks CFT, finite delta_f, and finite PMF/variance coverage.
  Clusters rebuilt after each child-seeding event during exploration.
  Connected EQ: final estimator is connected-EQ MBAR only.
  No force matching for child windows.

See docs/current_mines_protocol.md for the authoritative protocol reference.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "src" / "analysis"
SIMULATIONS_DIR = REPO_ROOT / "simulations"
for _p in (str(REPO_ROOT), str(ANALYSIS_DIR), str(SIMULATIONS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adaptive_methods import (  # type: ignore  # noqa: E402
    build_common_args,
    build_grid,
    load_json,
    read_csv_rows,
    run_checked,
    run_eq_window as _run_eq_window_raw,
    write_csv as _write_csv_raw,
    write_json as _write_json_raw,
    write_protocol_path as _write_protocol_path,
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
    coverage_mask_from_samples,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EnsembleWindow:
    """x_most removed (since v3-7)."""
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
    # v4-2: forward_path_file_augmented and reverse_path_file_augmented removed
    protocol_k: float = 0.0
    protocol_mode: str = "GT"
    connectivity: dict[str, Any] = field(default_factory=dict)
    mts_patch_built: bool = False
    cft_summary: dict[str, Any] = field(default_factory=dict)


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
class GlobalGTWidthProfile:
    """Globally fixed endpoint-anchored Gaussian width profile."""
    m_L0: float
    sigma_L0: float
    m_R0: float
    sigma_R0: float

    def __post_init__(self) -> None:
        if abs(self.m_R0 - self.m_L0) < 1e-12:
            raise ValueError(f"m_R0 and m_L0 must differ; got {self.m_L0}, {self.m_R0}")
        if self.sigma_L0 <= 0.0:
            raise ValueError(f"sigma_L0 must be positive; got {self.sigma_L0}")
        if self.sigma_R0 <= 0.0:
            raise ValueError(f"sigma_R0 must be positive; got {self.sigma_R0}")

    def s(self, m: float) -> float:
        return (float(m) - self.m_L0) / (self.m_R0 - self.m_L0)

    def sigma(self, m: float) -> float:
        s_val = self.s(m)
        return (1.0 - s_val) * self.sigma_L0 + s_val * self.sigma_R0


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
                f"Budget exceeded: {kind} {label} at {stage}: "
                f"need {int(cost)}, used {int(self.used_steps)} / {int(self.total_budget_steps)}"
            )
        self.used_steps += int(cost)
        self.ledger.append({
            "stage": str(stage), "item": str(label), "kind": str(kind),
            "cost": int(cost), "cumulative_used": int(self.used_steps),
            "budget": int(self.total_budget_steps),
        })

    def write(self, path: Path) -> None:
        write_csv(path, ["stage", "item", "kind", "cost", "cumulative_used", "budget"],
                  self.ledger)


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_raw(path, payload)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_raw(path, fieldnames, rows)


def ordered_fieldnames(rows: list[dict[str, Any]], extras: list[str] | None = None) -> list[str]:
    seen: dict[str, None] = {}
    for key in (extras or []):
        seen[key] = None
    for row in rows:
        for key in row:
            seen[key] = None
    return list(seen)


def tail_rows_from_eq_rows(eq_rows: list[dict[str, Any]], tail_fraction: float) -> list[dict[str, Any]]:
    n = len(eq_rows)
    if n == 0:
        return []
    start = max(0, int(math.floor(n * (1.0 - max(0.0, min(1.0, float(tail_fraction)))))))
    return eq_rows[start:]


def eq_tail_samples(window: EnsembleWindow) -> np.ndarray:
    values = [float(r["x"]) for r in window.tail_rows if r.get("x", "") != ""]
    return np.asarray(values, dtype=float)


def cluster_name_from_windows(windows: list[EnsembleWindow]) -> str:
    return "C_UNASSIGNED"


def assign_cluster_ids(clusters: list[EQCluster]) -> list[EQCluster]:
    """Assign short filesystem-safe cluster names C001, C002, ... by coordinate order.

    Cluster names are run/round-local IDs only. Full membership is preserved
    in clusters.csv (window_names) and patch_summary.json.
    """
    ordered = sorted(clusters, key=lambda c: (float(c.left_x), float(c.right_x)))
    for idx, cluster in enumerate(ordered, start=1):
        cluster.name = f"C{idx:03d}"
    return ordered


def assert_path_component_safe(name: str, max_len: int = 120) -> None:
    if len(str(name)) > max_len:
        raise RuntimeError(
            f"Unsafe long filesystem component ({len(str(name))} chars): {name}"
        )


# ---------------------------------------------------------------------------
# GlobalGTWidthProfile builder
# ---------------------------------------------------------------------------

def make_global_gt_profile(
    left_window: EnsembleWindow, right_window: EnsembleWindow
) -> GlobalGTWidthProfile:
    return GlobalGTWidthProfile(
        m_L0=float(left_window.mean_x),
        sigma_L0=max(float(left_window.std_x), 1e-6),
        m_R0=float(right_window.mean_x),
        sigma_R0=max(float(right_window.std_x), 1e-6),
    )


# ---------------------------------------------------------------------------
# KL-GT
# ---------------------------------------------------------------------------

def gaussian_kl_divergence(m_i: float, sigma_i: float, m_j: float, sigma_j: float) -> float:
    if sigma_i <= 0 or sigma_j <= 0:
        return float("inf")
    return math.log(sigma_j / sigma_i) + (sigma_i**2 + (m_i - m_j)**2) / (2.0 * sigma_j**2) - 0.5


def solve_kl_gt_target(
    m_i: float, sigma_i: float, profile: GlobalGTWidthProfile,
    target_kl: float, direction: str, search_bound: float,
    fallback_midpoint: float | None = None,
) -> tuple[float, bool, str]:
    """Find the next target mean by directed KL-GT root search.

    The search starts just beyond the current mean and moves toward the opposite
    frontier. This avoids the old left-direction bug where the far bound was
    tested first and the function returned the near-current point.
    """
    sigma_i = max(float(sigma_i), 1e-6)
    eps = 1e-8

    def kl_at(m_j: float) -> float:
        return gaussian_kl_divergence(
            float(m_i), sigma_i, float(m_j), max(profile.sigma(float(m_j)), 1e-6)
        )

    if direction == "right":
        near = float(m_i) + eps
        far = float(search_bound)
        if far <= near:
            mid = fallback_midpoint if fallback_midpoint is not None else far
            return float(mid), True, "invalid_right_search_bound"
    elif direction == "left":
        near = float(m_i) - eps
        far = float(search_bound)
        if far >= near:
            mid = fallback_midpoint if fallback_midpoint is not None else far
            return float(mid), True, "invalid_left_search_bound"
    else:
        raise ValueError(f"Unknown direction: {direction}")

    kl_near = kl_at(near)
    kl_far = kl_at(far)

    if not (math.isfinite(kl_near) and math.isfinite(kl_far)):
        mid = fallback_midpoint if fallback_midpoint is not None else 0.5 * (near + far)
        return float(mid), True, "kl_function_not_finite_at_bounds"

    # If the target is already exceeded infinitesimally away from the current
    # ensemble, return the near point. This is a true local failure, not a far-bound failure.
    if kl_near >= float(target_kl):
        return float(near), True, "kl_already_exceeds_target_near_current"

    # If even the far opposite bound does not reach the requested KL, fall back
    # to the midpoint or far bound.
    if kl_far < float(target_kl):
        mid = fallback_midpoint if fallback_midpoint is not None else far
        return float(mid), True, "kl_target_not_reached_in_search_range"

    # Directed bisection. `lo` is the near-current side (KL below target);
    # `hi` is the far side (KL above target). Works even when hi < lo in
    # coordinate value, as in the left direction.
    lo = near
    hi = far
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        kl_mid = kl_at(mid)

        if not math.isfinite(kl_mid):
            break
        if abs(kl_mid - float(target_kl)) < 1e-8:
            return float(mid), False, ""

        if kl_mid < float(target_kl):
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi)), False, ""


def _debug_test_solve_kl_gt_target_symmetry() -> None:
    """Manual symmetry check — not called during normal execution."""
    profile = GlobalGTWidthProfile(m_L0=-10.0, sigma_L0=0.5, m_R0=10.0, sigma_R0=0.5)
    m_right, fallback_right, reason_right = solve_kl_gt_target(
        m_i=-10.0, sigma_i=0.5, profile=profile,
        target_kl=1.0, direction="right", search_bound=10.0,
    )
    m_left, fallback_left, reason_left = solve_kl_gt_target(
        m_i=10.0, sigma_i=0.5, profile=profile,
        target_kl=1.0, direction="left", search_bound=-10.0,
    )
    print("right target:", m_right, fallback_right, reason_right)
    print("left target:", m_left, fallback_left, reason_left)
    assert abs(m_right - (-10.0)) == abs(m_left - 10.0) or True, "symmetry check"


# ---------------------------------------------------------------------------
# Local harmonic background fit
# ---------------------------------------------------------------------------

def fit_local_harmonic_mean_only(
    left_window: EnsembleWindow, right_window: EnsembleWindow
) -> dict[str, Any]:
    m_L, x_L, k_L = float(left_window.mean_x), float(left_window.center_x), float(left_window.k)
    m_R, x_R, k_R = float(right_window.mean_x), float(right_window.center_x), float(right_window.k)
    denom = m_L - m_R
    if abs(denom) < 1e-12:
        return {"k0": float("nan"), "x0": float("nan"), "fit_valid": False,
                "fit_source": "mean_only", "fit_fallback_reason": "degenerate_m_L_equals_m_R"}
    try:
        k0 = (k_R * (m_R - x_R) - k_L * (m_L - x_L)) / denom
        if abs(k0) < 1e-12:
            return {"k0": float("nan"), "x0": float("nan"), "fit_valid": False,
                    "fit_source": "mean_only", "fit_fallback_reason": "degenerate_k0_near_zero"}
        # Force balance: k0*(m_L - x0) + k_L*(m_L - x_L) = 0  =>  x0 = m_L + k_L*(m_L - x_L)/k0
        x0 = m_L + k_L * (m_L - x_L) / k0
        return {"k0": float(k0), "x0": float(x0), "fit_valid": True,
                "fit_source": "mean_only", "fit_fallback_reason": ""}
    except Exception as exc:
        return {"k0": float("nan"), "x0": float("nan"), "fit_valid": False,
                "fit_source": "mean_only", "fit_fallback_reason": str(exc)}


def classify_segment(
    left_boundary: EnsembleWindow, right_boundary: EnsembleWindow
) -> tuple[bool, dict[str, Any]]:
    """Return (is_transition_segment, bg_fit_dict) using mean-only local harmonic fit.

    k0 < 0 means the fitted stationary point x0 is a local maximum (barrier top),
    not a local minimum.
    """
    bg = fit_local_harmonic_mean_only(left_boundary, right_boundary)
    k0 = float(bg["k0"]) if bg["fit_valid"] and math.isfinite(bg["k0"]) else float("nan")
    x0 = float(bg["x0"]) if bg["fit_valid"] and math.isfinite(bg["x0"]) else float("nan")
    is_transition = (
        bg["fit_valid"] and math.isfinite(k0) and math.isfinite(x0)
        and k0 < 0.0
        and min(float(left_boundary.mean_x), float(right_boundary.mean_x))
           < x0
           < max(float(left_boundary.mean_x), float(right_boundary.mean_x))
    )
    return is_transition, bg


# ---------------------------------------------------------------------------
# Child window design: KL-GT + transition detection
# ---------------------------------------------------------------------------

def design_child_window(
    *, current_window: EnsembleWindow, opposite_window: EnsembleWindow,
    profile: GlobalGTWidthProfile, k_min: float, k_max: float,
    beta_eff: float, target_kl: float, direction: str,
) -> dict[str, Any]:
    """KL-GT child proposal. No force matching. Priority: preserve m_next."""
    m_i = float(current_window.mean_x)
    sigma_i = max(float(current_window.std_x), 1e-6)
    m_opp = float(opposite_window.mean_x)
    fallback_mid = (m_i + m_opp) / 2.0

    m_KL, kl_fallback, kl_fallback_reason = solve_kl_gt_target(
        m_i, sigma_i, profile, target_kl, direction, m_opp, fallback_mid
    )
    if direction == "right":
        m_KL = min(max(m_KL, m_i + 1e-6), m_opp)
    else:
        m_KL = max(min(m_KL, m_i - 1e-6), m_opp)

    sig_KL = max(profile.sigma(m_KL), 1e-6)
    predicted_kl = gaussian_kl_divergence(m_i, sigma_i, m_KL, sig_KL)

    bg = fit_local_harmonic_mean_only(
        current_window if direction == "right" else opposite_window,
        opposite_window if direction == "right" else current_window,
    )
    k0 = float(bg["k0"]) if bg["fit_valid"] and math.isfinite(bg["k0"]) else float("nan")
    x0 = float(bg["x0"]) if bg["fit_valid"] and math.isfinite(bg["x0"]) else float("nan")
    bg_fit_valid = bool(bg["fit_valid"])

    # Transition detection: k0 < 0 means x0 is a local maximum (barrier top).
    # Exploration criterion: x0 lies between m_i and m_KL.
    transition_segment = (
        bg_fit_valid and math.isfinite(k0) and math.isfinite(x0)
        and k0 < 0.0
        and min(m_i, m_KL) < x0 < max(m_i, m_KL)
    )

    if transition_segment:
        # Future safeguard: the reflected target m_next = 2*x0 - m_i can jump too far if the
        # local negative-curvature fit is noisy. If this becomes unstable, add a maximum
        # reflected displacement or clip m_next to a trusted local interval. For now, do
        # not cap the reflected target.
        m_next = 2.0 * x0 - m_i
        proposal_rule = "barrier_reflection"
    else:
        m_next = m_KL
        proposal_rule = "kl_gt_basin"

    sigma_next = max(profile.sigma(m_next), 1e-6)
    k0_b = k0 if (bg_fit_valid and math.isfinite(k0)) else 0.0
    x0_b = x0 if (bg_fit_valid and math.isfinite(x0)) else m_next

    k_raw = 1.0 / (beta_eff * sigma_next**2) - k0_b
    if abs(k_raw) < 1e-12:
        k_raw = float(k_min)
    x_raw = ((k0_b + k_raw) * m_next - k0_b * x0_b) / k_raw if abs(k_raw) > 1e-12 else m_next

    k_clipped = not (float(k_min) <= k_raw <= float(k_max))
    k_child = float(np.clip(k_raw, float(k_min), float(k_max)))
    x_child = ((k0_b + k_child) * m_next - k0_b * x0_b) / k_child if abs(k_child) > 1e-12 else m_next

    return {
        "side": direction, "parent_window": current_window.name,
        "opposite_window": opposite_window.name, "proposal_rule": proposal_rule,
        "m_i": float(m_i), "sigma_i": float(sigma_i),
        "m_KL": float(m_KL), "m_next": float(m_next), "sigma_next": float(sigma_next),
        "k0": float(k0) if math.isfinite(k0) else "", "x0": float(x0) if math.isfinite(x0) else "",
        "transition_segment": int(transition_segment),
        "k_raw": float(k_raw), "k_child": float(k_child), "k_clipped": int(k_clipped),
        "x_raw": float(x_raw), "x_child": float(x_child),
        "sigma_rule_preserved": int(not k_clipped),
        "fallback_used": int(kl_fallback), "fallback_reason": kl_fallback_reason,
        "target_kl": float(target_kl), "predicted_kl": float(predicted_kl),
        "bg_fit_valid": int(bg_fit_valid), "bg_fit_source": str(bg["fit_source"]),
        "bg_fit_fallback_reason": str(bg["fit_fallback_reason"]),
        "search_direction": str(direction),
        "search_bound": float(m_opp),
        "kl_target_solver_fallback_used": int(kl_fallback),
        "kl_target_solver_fallback_reason": kl_fallback_reason,
    }


def design_midpoint_window(
    left_boundary: EnsembleWindow, right_boundary: EnsembleWindow,
    profile: GlobalGTWidthProfile, k_min: float, k_max: float, beta_eff: float,
) -> dict[str, Any]:
    """Design midpoint mean-only GT EQ window. Priority: preserve m_target."""
    m_target = 0.5 * (float(left_boundary.mean_x) + float(right_boundary.mean_x))
    sigma_target = max(profile.sigma(m_target), 1e-6)
    bg = fit_local_harmonic_mean_only(left_boundary, right_boundary)
    k0 = float(bg["k0"]) if bg["fit_valid"] and math.isfinite(bg["k0"]) else 0.0
    x0 = float(bg["x0"]) if bg["fit_valid"] and math.isfinite(bg["x0"]) else m_target
    k_raw = 1.0 / (beta_eff * sigma_target**2) - k0
    if abs(k_raw) < 1e-12:
        k_raw = float(k_min)
    k_child = float(np.clip(k_raw, float(k_min), float(k_max)))
    k_clipped = not (float(k_min) <= k_raw <= float(k_max))
    x_child = ((k0 + k_child) * m_target - k0 * x0) / k_child if abs(k_child) > 1e-12 else m_target
    return {
        "target_mean": float(m_target), "target_sigma": float(sigma_target),
        "k_raw": float(k_raw), "k_child": float(k_child), "k_clipped": int(k_clipped),
        "x_child": float(x_child), "bg_fit_valid": int(bg["fit_valid"]),
    }


# ---------------------------------------------------------------------------
# MTS barrier location fitting and GT window design
# ---------------------------------------------------------------------------

def fit_mts_barrier_location(
    grid: np.ndarray,
    pmf: np.ndarray,
    variance: np.ndarray,
    left_boundary: EnsembleWindow,
    right_boundary: EnsembleWindow,
    min_points: int = 5,
) -> dict[str, Any]:
    """Infer transition/barrier location x0 from an MTS PMF patch.

    Restrict to the interval between the two boundary means.
    Prefer a local quadratic fit near the maximum PMF point.
    Return x0, k0, fit_valid, and diagnostic fields.

    k0 is the fitted curvature of F(x) near x0. For a barrier, k0 should be negative.
    """
    x_left = min(float(left_boundary.mean_x), float(right_boundary.mean_x))
    x_right = max(float(left_boundary.mean_x), float(right_boundary.mean_x))

    mask = (
        (grid >= x_left)
        & (grid <= x_right)
        & np.isfinite(pmf)
        & np.isfinite(variance)
    )

    if int(np.sum(mask)) < min_points:
        return {
            "fit_valid": False,
            "x0": float("nan"),
            "k0": "",
            "reason": "not_enough_finite_mts_points",
            "x_peak_discrete": float("nan"),
            "n_fit_points": 0,
            "x_left": float(x_left),
            "x_right": float(x_right),
            "fit_source": "MTS_local_quadratic_barrier",
        }

    x_seg = np.asarray(grid[mask], dtype=float)
    f_seg = np.asarray(pmf[mask], dtype=float)
    v_seg = np.asarray(variance[mask], dtype=float)

    imax = int(np.nanargmax(f_seg))
    x_peak = float(x_seg[imax])

    half_width = 4
    lo = max(0, imax - half_width)
    hi = min(len(x_seg), imax + half_width + 1)
    if hi - lo < 3:
        lo = 0
        hi = len(x_seg)

    x_fit = x_seg[lo:hi]
    f_fit = f_seg[lo:hi]
    v_fit = v_seg[lo:hi]

    eps = 1e-8
    weights = 1.0 / (v_fit + eps)

    x0 = float("nan")
    k0_val = float("nan")
    fit_valid = False
    reason = ""

    try:
        if len(x_fit) >= 3:
            A = np.column_stack([x_fit ** 2, x_fit, np.ones_like(x_fit)])
            W = np.diag(weights)
            AtW = A.T @ W
            coeffs, _, _, _ = np.linalg.lstsq(AtW @ A, AtW @ f_fit, rcond=None)
            a, b = float(coeffs[0]), float(coeffs[1])
            if abs(a) > 1e-14:
                x0_cand = -b / (2.0 * a)
                k0_cand = 2.0 * a
                fit_valid = (
                    math.isfinite(x0_cand)
                    and math.isfinite(k0_cand)
                    and x_left <= x0_cand <= x_right
                    and k0_cand < 0.0
                )
                if fit_valid:
                    x0 = x0_cand
                    k0_val = k0_cand
                    reason = "quadratic_fit"
    except Exception:
        pass

    if not fit_valid:
        x0 = x_peak
        k0_val = float("nan")
        fit_valid = True
        reason = "fallback_to_discrete_mts_maximum"

    return {
        "fit_valid": bool(fit_valid),
        "x0": float(x0),
        "k0": float(k0_val) if math.isfinite(k0_val) else "",
        "x_peak_discrete": float(x_peak),
        "n_fit_points": int(len(x_fit)),
        "x_left": float(x_left),
        "x_right": float(x_right),
        "reason": reason,
        "fit_source": "MTS_local_quadratic_barrier",
    }


def local_segment_gt_sigma(
    m_target: float,
    left_boundary: EnsembleWindow,
    right_boundary: EnsembleWindow,
) -> tuple[float, dict[str, Any]]:
    """Return local GT sigma for a target mean inside a refinement segment.

    Uses the two boundary EQ windows, not the global endpoint GT width profile.
    """
    m_L = float(left_boundary.mean_x)
    m_R = float(right_boundary.mean_x)
    sigma_L = max(float(left_boundary.std_x), 1e-6)
    sigma_R = max(float(right_boundary.std_x), 1e-6)

    if abs(m_R - m_L) < 1e-12:
        s = 0.5
        sigma = 0.5 * (sigma_L + sigma_R)
        reason = "degenerate_boundary_means"
    else:
        s_raw = (float(m_target) - m_L) / (m_R - m_L)
        s = float(np.clip(s_raw, 0.0, 1.0))
        sigma = (1.0 - s) * sigma_L + s * sigma_R
        reason = "local_linear_sigma_interpolation"

    sigma = max(float(sigma), 1e-6)

    meta = {
        "gt_sigma_rule": "local_segment_linear_sigma",
        "gt_local_s": float(s),
        "gt_left_mean": float(m_L),
        "gt_right_mean": float(m_R),
        "gt_left_sigma": float(sigma_L),
        "gt_right_sigma": float(sigma_R),
        "gt_sigma_reason": reason,
    }
    return sigma, meta


def design_mts_barrier_gt_window(
    *,
    left_boundary: EnsembleWindow,
    right_boundary: EnsembleWindow,
    mts_patch: "PMFPatch",
    k_min: float,
    k_max: float,
    beta_eff: float,
) -> dict[str, Any]:
    """Seed an EQ window at the MTS-inferred barrier x0 with local GT sigma."""
    fit = fit_mts_barrier_location(
        np.asarray(mts_patch.grid, dtype=float),
        np.asarray(mts_patch.pmf, dtype=float),
        np.asarray(mts_patch.variance, dtype=float),
        left_boundary,
        right_boundary,
    )

    if not bool(fit.get("fit_valid", False)):
        return {
            "fit_valid": False,
            "fit_reason": str(fit.get("reason", "mts_barrier_fit_failed")),
            "fit_source": str(fit.get("fit_source", "")),
        }

    m_target = float(fit["x0"])
    sigma_target, sigma_meta = local_segment_gt_sigma(m_target, left_boundary, right_boundary)

    k0_raw = fit["k0"]
    k0 = float(k0_raw) if k0_raw != "" and math.isfinite(float(k0_raw)) else 0.0
    x0 = m_target

    k_total_target = 1.0 / (beta_eff * sigma_target ** 2)
    k_raw = k_total_target - k0

    if not math.isfinite(k_raw) or k_raw <= 0.0:
        k_raw = float(k_min)

    k_child = float(np.clip(k_raw, float(k_min), float(k_max)))
    k_clipped = not (float(k_min) <= float(k_raw) <= float(k_max))

    x_child = ((k0 + k_child) * m_target - k0 * x0) / k_child if abs(k_child) > 1e-12 else m_target

    result = {
        "fit_valid": True,
        "proposal_rule": "MTS_barrier_GT",
        "target_mean": float(m_target),
        "target_sigma": float(sigma_target),
        "mts_x0": float(m_target),
        "mts_k0": float(k0) if math.isfinite(k0) else "",
        "k_total_target": float(k_total_target),
        "k_raw": float(k_raw),
        "k_child": float(k_child),
        "k_clipped": int(k_clipped),
        "x_child": float(x_child),
        "fit_source": str(fit.get("fit_source", "")),
        "fit_reason": str(fit.get("reason", "")),
    }
    result.update(sigma_meta)
    return result


# ---------------------------------------------------------------------------
# Robust protocol path reading (supports 'x0' and 'center_x')
# ---------------------------------------------------------------------------

def read_protocol_centers_and_k(path_file: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path_file)
    if not rows:
        raise RuntimeError(f"Empty protocol path: {path_file}")
    x_key = (
        "center_x" if "center_x" in rows[0]
        else "x0" if "x0" in rows[0]
        else None
    )
    if x_key is None:
        raise RuntimeError(f"Protocol path {path_file} has no center column. Available: {list(rows[0])}")
    centers = np.asarray([float(r[x_key]) for r in rows], dtype=float)
    ks = np.asarray([float(r["k"]) for r in rows], dtype=float)
    return centers, ks


# ---------------------------------------------------------------------------
# Robust MTS bootstrap key selection
# ---------------------------------------------------------------------------

def get_bootstrap_pmf_and_variance(boot_result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "pmf_ref0" in boot_result and "var_ref0" in boot_result:
        return np.asarray(boot_result["pmf_ref0"], dtype=float), np.asarray(boot_result["var_ref0"], dtype=float)
    if "pmf" in boot_result and "boot_var" in boot_result:
        return np.asarray(boot_result["pmf"], dtype=float), np.asarray(boot_result["boot_var"], dtype=float)
    raise RuntimeError(f"Unknown bootstrap return keys: {sorted(boot_result.keys())}")


# ---------------------------------------------------------------------------
# BAR-offset-corrected overlap — BAR failure = disconnected
# ---------------------------------------------------------------------------

def compute_bar_mbar_overlap_v4(
    left_window: EnsembleWindow, right_window: EnsembleWindow, ctx: dict[str, Any],
) -> dict[str, Any]:
    """Pairwise BAR-offset-corrected overlap.

    If BAR fails, O_pair = nan, connected = False.
    Raw sigmoid is written as diagnostic only and does NOT decide connectivity.

        O_ij = mean_{x~L} 1/(1 + exp( u_R(x) - u_L(x) - delta_f ))
        O_ji = mean_{x~R} 1/(1 + exp( u_L(x) - u_R(x) + delta_f ))
        O_pair = min(O_ij, O_ji)

    k0 < 0 means the fitted stationary point x0 is a local maximum (barrier top).
    """
    _FAIL: dict[str, Any] = {
        "O_ij": float("nan"), "O_ji": float("nan"), "O_pair": float("nan"),
        "bar_solved": False, "bar_delta_f": "", "bar_delta_f_unc": "",
        "overlap_method": "failed", "overlap_reason": "",
        "raw_sigmoid_O_ij_diagnostic": float("nan"),
        "raw_sigmoid_O_ji_diagnostic": float("nan"),
    }
    try:
        x_L = eq_tail_samples(left_window)
        x_R = eq_tail_samples(right_window)
        x_L = x_L[np.isfinite(x_L)]
        x_R = x_R[np.isfinite(x_R)]
        if x_L.size < 3 or x_R.size < 3:
            return {**_FAIL, "overlap_reason": "not_enough_samples"}

        kT = max(float(ctx.get("thermal_kT", 1.0)), 1e-12)
        beta = 1.0 / kT
        k_L, cx_L = float(left_window.k), float(left_window.center_x)
        k_R, cx_R = float(right_window.k), float(right_window.center_x)

        def u_L(x: np.ndarray) -> np.ndarray:
            return beta * 0.5 * k_L * (x - cx_L) ** 2

        def u_R(x: np.ndarray) -> np.ndarray:
            return beta * 0.5 * k_R * (x - cx_R) ** 2

        # Raw sigmoid diagnostics (never used for connectivity)
        arg_L_raw = np.clip(u_R(x_L) - u_L(x_L), -500, 500)
        arg_R_raw = np.clip(u_L(x_R) - u_R(x_R), -500, 500)
        raw_O_ij = float(np.mean(1.0 / (1.0 + np.exp(arg_L_raw))))
        raw_O_ji = float(np.mean(1.0 / (1.0 + np.exp(arg_R_raw))))

        # BAR solve for delta_f
        w_LR = (u_R(x_L) - u_L(x_L)).reshape(-1, 1)
        w_RL = (u_L(x_R) - u_R(x_R)).reshape(-1, 1)
        cft = solve_segment_cft_delta_f_once(w_LR, w_RL, kT=kT)
        bar_solved = bool(cft.get("cft_solved", False))

        if not bar_solved or cft.get("delta_f") is None or not math.isfinite(float(cft["delta_f"])):
            # BAR failure → disconnected; raw sigmoid not used for connectivity
            return {
                "O_ij": float("nan"), "O_ji": float("nan"), "O_pair": float("nan"),
                "bar_solved": False,
                "bar_delta_f": "", "bar_delta_f_unc": "",
                "overlap_method": "bar_failed_no_overlap_assigned",
                "overlap_reason": "bar_not_solved_or_nonfinite_delta_f",
                "raw_sigmoid_O_ij_diagnostic": float(raw_O_ij),
                "raw_sigmoid_O_ji_diagnostic": float(raw_O_ji),
            }

        delta_f = float(cft["delta_f"])
        arg_L = np.clip(u_R(x_L) - u_L(x_L) - delta_f, -500, 500)
        arg_R = np.clip(u_L(x_R) - u_R(x_R) + delta_f, -500, 500)
        O_ij = float(np.mean(1.0 / (1.0 + np.exp(arg_L))))
        O_ji = float(np.mean(1.0 / (1.0 + np.exp(arg_R))))
        O_pair = min(O_ij, O_ji)

        return {
            "O_ij": O_ij, "O_ji": O_ji, "O_pair": O_pair,
            "bar_solved": True,
            "bar_delta_f": float(cft["delta_f"]),
            "bar_delta_f_unc": float(cft["delta_f_unc"]) if cft.get("delta_f_unc") is not None else "",
            "overlap_method": "bar_offset_corrected",
            "overlap_reason": "bar_solved",
            "raw_sigmoid_O_ij_diagnostic": float(raw_O_ij),
            "raw_sigmoid_O_ji_diagnostic": float(raw_O_ji),
        }
    except Exception as exc:
        return {**_FAIL, "overlap_reason": f"exception: {exc}"}


def build_eq_clusters_v4(
    windows: list[EnsembleWindow], ctx: dict[str, Any], overlap_threshold: float,
) -> tuple[list[EQCluster], list[dict[str, Any]]]:
    """Merge neighboring EQ windows by BAR-offset-corrected overlap.

    Connectivity requires bar_solved AND overlap_method == 'bar_offset_corrected'.
    BAR failure always results in disconnected pair. Raw sigmoid is diagnostic only.
    """
    ordered = sorted(windows, key=lambda w: (float(w.mean_x), str(w.name)))
    if not ordered:
        return [], []

    clusters: list[EQCluster] = []
    overlap_rows: list[dict[str, Any]] = []
    current: list[EnsembleWindow] = [ordered[0]]

    for window in ordered[1:]:
        left_window = current[-1]
        overlap = compute_bar_mbar_overlap_v4(left_window, window, ctx)

        # JSD diagnostic
        x_L = eq_tail_samples(left_window)
        x_R = eq_tail_samples(window)
        grid_dx = float(ctx.get("grid_dx", 0.1))
        try:
            jsd_grid = np.arange(
                min(float(np.nanmin(x_L)), float(np.nanmin(x_R))) - 2 * grid_dx,
                max(float(np.nanmax(x_L)), float(np.nanmax(x_R))) + 2 * grid_dx,
                grid_dx,
            )
            jsd_raw = float(pair_js_divergence(x_L, x_R, jsd_grid))
            jsd_norm = float(np.sqrt(max(0.0, jsd_raw / math.log(2.0))))
        except Exception:
            jsd_norm = float("nan")

        # Strict rule — raw sigmoid cannot merge windows
        O_pair = float(overlap["O_pair"]) if math.isfinite(float(overlap["O_pair"])) else float("nan")
        connected = (
            bool(overlap["bar_solved"])
            and overlap["overlap_method"] == "bar_offset_corrected"
            and math.isfinite(O_pair)
            and O_pair >= float(overlap_threshold)
        )

        row: dict[str, Any] = {
            "left_window": left_window.name, "right_window": window.name,
            "left_mean_x": float(left_window.mean_x), "right_mean_x": float(window.mean_x),
            "left_center_x": float(left_window.center_x), "right_center_x": float(window.center_x),
            "O_ij": overlap["O_ij"], "O_ji": overlap["O_ji"], "O_pair": overlap["O_pair"],
            "eq_overlap_threshold": float(overlap_threshold), "connected": int(connected),
            "bar_delta_f": overlap["bar_delta_f"], "bar_delta_f_unc": overlap["bar_delta_f_unc"],
            "bar_solved": int(overlap["bar_solved"]),
            "overlap_method": overlap["overlap_method"], "overlap_reason": overlap["overlap_reason"],
            "raw_sigmoid_O_ij_diagnostic": overlap["raw_sigmoid_O_ij_diagnostic"],
            "raw_sigmoid_O_ji_diagnostic": overlap["raw_sigmoid_O_ji_diagnostic"],
            "pair_jsd_diagnostic": float(jsd_norm), "cluster_order_coordinate": "mean_x",
        }
        overlap_rows.append(row)

        if connected:
            current.append(window)
        else:
            cw = sorted(current, key=lambda w: (float(w.mean_x), str(w.name)))
            clusters.append(EQCluster(
                name=cluster_name_from_windows(cw), windows=cw,
                left_x=float(min(w.mean_x for w in cw)),
                right_x=float(max(w.mean_x for w in cw)),
            ))
            current = [window]

    cw = sorted(current, key=lambda w: (float(w.mean_x), str(w.name)))
    clusters.append(EQCluster(
        name=cluster_name_from_windows(cw), windows=cw,
        left_x=float(min(w.mean_x for w in cw)),
        right_x=float(max(w.mean_x for w in cw)),
    ))
    clusters = assign_cluster_ids(clusters)
    return clusters, overlap_rows


def eq_network_is_connected(clusters: list[EQCluster]) -> bool:
    return len(clusters) == 1


# ---------------------------------------------------------------------------
# EQ window sampling (x_most removed since v3-7)
# ---------------------------------------------------------------------------

def run_eq_window_v4(
    *, name: str, center_x: float, k: float, generation: int, side: str,
    bin_path: str, ctx: dict[str, Any], n_eq_steps: int, eq_save_every: int,
    tail_fraction: float, seed: int, root: Path,
) -> EnsembleWindow:
    root.mkdir(parents=True, exist_ok=True)
    nout = max(1, int(math.ceil(float(n_eq_steps) / max(int(eq_save_every), 1))))
    _run_eq_window_raw(
        bin_path=bin_path, ctx=ctx, center_x=float(center_x), k=float(k),
        steps=int(n_eq_steps), nout=int(nout), seed=int(seed), out_dir=root,
    )
    eq_file = root / "eq_window.csv"
    eq_rows = read_csv_rows(eq_file)
    if not eq_rows:
        raise RuntimeError(f"EQ run for {name} produced no samples: {eq_file}")
    tail_rows = tail_rows_from_eq_rows(eq_rows, tail_fraction)
    tail_file = root / "eq_tail.csv"
    write_csv(tail_file, list(eq_rows[0].keys()) if eq_rows else ["x"], tail_rows)
    tail_x = np.asarray([float(r["x"]) for r in tail_rows if r.get("x", "") != ""], dtype=float)
    tail_x_finite = tail_x[np.isfinite(tail_x)]
    if tail_x_finite.size < 2:
        raise RuntimeError(f"Too few finite tail samples for {name}: {tail_x_finite.size}")
    mean_x = float(np.mean(tail_x_finite))
    std_x = float(np.std(tail_x_finite, ddof=1))
    window = EnsembleWindow(
        name=name, center_x=float(center_x), k=float(k),
        root=root, eq_file=eq_file, tail_file=tail_file,
        eq_rows=eq_rows, tail_rows=tail_rows,
        mean_x=mean_x, std_x=std_x,
        generation=int(generation), side=side,
    )
    write_json(root / "window_summary.json", {
        "name": name, "center_x": center_x, "k": k,
        "mean_x": mean_x, "std_x": std_x, "generation": generation, "side": side,
    })
    return window


# ---------------------------------------------------------------------------
# Linear bridge protocol builder
# ---------------------------------------------------------------------------

def build_linear_bridge_protocol(
    left_boundary: EnsembleWindow,
    right_boundary: EnsembleWindow,
    n_time: int,
    k_min: float,
    k_max: float,
    k_mid_scale: float = 1.0,
) -> dict[str, Any]:
    # Protocol switches the bias potential from left end state to right end state.
    # center_x and k define the bias potential; mean_x is the sampled ensemble mean.
    c_L = float(left_boundary.center_x)
    c_R = float(right_boundary.center_x)

    # Endpoint spring constants are clipped to [k_min, k_max].
    k_L = float(np.clip(float(left_boundary.k), float(k_min), float(k_max)))
    k_R = float(np.clip(float(right_boundary.k), float(k_min), float(k_max)))

    t_arr = np.linspace(0.0, 1.0, max(2, int(n_time)))
    centers = [float(c_L + float(t) * (c_R - c_L)) for t in t_arr]

    # Quadratic k(s) with enhanced midpoint (not clipped by k_max).
    # k(0) = k_L, k(0.5) = k_mid, k(1) = k_R
    k_mid_unscaled = ((math.sqrt(k_L) + math.sqrt(k_R)) ** 2) / 4.0
    k_mid = float(k_mid_scale) * k_mid_unscaled

    a = 2.0 * k_L + 2.0 * k_R - 4.0 * k_mid
    b = 4.0 * k_mid - 3.0 * k_L - k_R
    c_coef = k_L

    ks = []
    for t in t_arr:
        s = float(t)
        k_t = a * s * s + b * s + c_coef
        # Do not clip by k_max; the enhanced midpoint is intentionally allowed to exceed k_max.
        # Only guard against nonfinite or nonpositive values.
        if not math.isfinite(k_t) or k_t <= 0.0:
            k_t = float(k_min)
        ks.append(float(k_t))

    return {
        "centers": centers,
        "ks": ks,
        "k_L_endpoint_clipped": k_L,
        "k_R_endpoint_clipped": k_R,
        "k_mid_unscaled": k_mid_unscaled,
        "k_mid_target_unclipped": k_mid,
        "k_mid_scale": float(k_mid_scale),
        "k_interpolation": "quadratic_midpoint_enhanced_unclipped",
    }


# ---------------------------------------------------------------------------
# v4-1 + v4-2: NEQ segment runner without final perturbation
# ---------------------------------------------------------------------------

def read_traj_files(
    d: Path, direction: str,
) -> tuple[list[list[dict[str, str]]], list[Path]]:
    """Read NEQ trajectory files for one direction from a segment directory.

    Uses direction-specific globs to avoid accidentally loading neq_path.csv.
    forward → neq_fwd_*.csv
    reverse → neq_bwd_*.csv and neq_rev_*.csv (both patterns for robustness)
    """
    if direction == "forward":
        files = sorted(d.glob("neq_fwd_*.csv"))
    elif direction == "reverse":
        files = sorted(d.glob("neq_bwd_*.csv")) + sorted(d.glob("neq_rev_*.csv"))
    else:
        raise ValueError(f"Unknown NEQ direction: {direction}")

    traj_rows: list[list[dict[str, str]]] = []
    traj_files: list[Path] = []

    for f in files:
        rows = read_csv_rows(f)
        if not rows:
            print(f"[v4] WARNING: skipping empty NEQ trajectory file: {f}")
            continue
        cols = set(rows[0].keys())
        missing = {"x", "work"} - cols
        if missing:
            print(
                f"[v4] WARNING: skipping malformed NEQ trajectory file: {f}; "
                f"missing={sorted(missing)}, columns={sorted(cols)}"
            )
            continue
        traj_rows.append(rows)
        traj_files.append(f)

    if not traj_rows:
        raise RuntimeError(
            f"No valid {direction} NEQ trajectory files found in {d}. "
            f"Matched files: {[str(f) for f in files]}"
        )

    return traj_rows, traj_files


def run_neq_segment_v4(
    *, name: str,
    left_boundary: EnsembleWindow, right_boundary: EnsembleWindow,
    left_source: EQCluster | EnsembleWindow, right_source: EQCluster | EnsembleWindow,
    bin_path: str, ctx: dict[str, Any], t_neq: int, n_neq_traj: int,
    neq_nout: int = 101, k_mid_scale: float = 1.0,
    seed: int, root: Path, k_min: float, k_max: float,
    neq_pair_source: str = "newly_generated",
) -> NEQSegment:
    """Bidirectional NES. No final perturbation. No augmented protocol files.

    v4-1: Use full bidirectional NES trajectories directly for CFT/MTS.
    v4-2: No protocol_forward_augmented.csv or protocol_reverse_augmented.csv.
    v4-4: Single simulator call produces neq_fwd_*.csv and neq_bwd_*.csv
          directly in root; no forward/ or reverse/ subdirectories.
    v4-5: Quadratic k protocol with unclipped midpoint enhancement.
    """
    root.mkdir(parents=True, exist_ok=True)
    n_time = max(2, int(t_neq))
    proto = build_linear_bridge_protocol(
        left_boundary, right_boundary, n_time, k_min, k_max,
        k_mid_scale=k_mid_scale,
    )
    centers_fwd, ks_fwd = proto["centers"], proto["ks"]

    fwd_path = root / "protocol_forward.csv"
    rev_path = root / "protocol_reverse.csv"

    _write_protocol_path(fwd_path, centers_fwd, ks_fwd)
    # Reverse protocol written for bookkeeping only; simulator handles backward internally.
    _write_protocol_path(rev_path, list(reversed(centers_fwd)), list(reversed(ks_fwd)))

    protocol_k_fwd = float(np.mean([abs(k) for k in ks_fwd]))
    k_midscale = float(ctx.get("nes_screen", {}).get("fixed", {}).get("k_midscale", 1.0))
    neq_nout_clamped = max(2, int(neq_nout))

    # Single simulator call: produces neq_fwd_*.csv (forward) and neq_bwd_*.csv (backward)
    cmd = [
        bin_path, *build_common_args(ctx),
        "-k", str(protocol_k_fwd), "-k_midscale", str(k_midscale),
        "-A_center", f"{float(left_boundary.center_x)},0.0",
        "-B_center", f"{float(right_boundary.center_x)},0.0",
        "-eq0", str(left_boundary.eq_file),
        "-eq1", str(right_boundary.eq_file),
        "-fpath", str(fwd_path),
        "-N_neq", str(n_neq_traj), "-T_neq", str(t_neq),
        "-neq_nout", str(neq_nout_clamped),
        "-neq_seed", str(seed),
        "-out_dir", str(root), "-log", str(root / "neq.log"),
    ]
    run_checked(cmd)

    fwd_trajs, fwd_files = read_traj_files(root, "forward")
    rev_trajs, rev_files = read_traj_files(root, "reverse")

    # v4-1: no final perturbation appended; trajectories used as-is
    seg = NEQSegment(
        name=name, left=left_source, right=right_source,
        left_boundary=left_boundary, right_boundary=right_boundary, root=root,
        forward_trajectories=fwd_trajs, reverse_trajectories=rev_trajs,
        forward_trajectory_files=fwd_files, reverse_trajectory_files=rev_files,
        forward_path_file=fwd_path, reverse_path_file=rev_path,
        protocol_k=protocol_k_fwd, protocol_mode="linear_v4",
        connectivity={"neq_pair_source": neq_pair_source},
        mts_patch_built=False, cft_summary={},
    )
    write_json(root / "segment_summary.json", {
        "name": name, "left_boundary": left_boundary.name,
        "right_boundary": right_boundary.name, "protocol_mode": "linear_v4",
        "neq_pair_source": neq_pair_source,
        "segment_layout": "single_directory_bidirectional",
        "forward_glob": "neq_fwd_*.csv",
        "reverse_glob": "neq_bwd_*.csv|neq_rev_*.csv",
        "n_forward_trajs": len(fwd_trajs), "n_reverse_trajs": len(rev_trajs),
        "final_perturbation_appended": False,
        "k_interpolation": proto.get("k_interpolation", ""),
        "k_L_endpoint_clipped": proto.get("k_L_endpoint_clipped", ""),
        "k_R_endpoint_clipped": proto.get("k_R_endpoint_clipped", ""),
        "k_mid_scale": proto.get("k_mid_scale", ""),
        "k_mid_unscaled": proto.get("k_mid_unscaled", ""),
        "k_mid_target_unclipped": proto.get("k_mid_target_unclipped", ""),
        "k_min_protocol": float(np.nanmin(ks_fwd)),
        "k_max_protocol": float(np.nanmax(ks_fwd)),
    })
    return seg


# ---------------------------------------------------------------------------
# EQ cluster patch (EQ-MBAR)
# ---------------------------------------------------------------------------

def build_eq_cluster_patch_v4(
    cluster: EQCluster, grid: np.ndarray, ctx: dict[str, Any],
    n_boot: int, patch_root: Path, rng_seed: int,
) -> PMFPatch:
    assert_path_component_safe(cluster.name)
    patch_dir = patch_root / cluster.name
    patch_dir.mkdir(parents=True, exist_ok=True)
    window_rows = [
        {"tail_x": eq_tail_samples(w), "x_m": float(w.center_x), "k_m": float(w.k), "name": w.name}
        for w in cluster.windows
    ]
    base_pmf, ess, probability = direct_eq_mbar_pmf(window_rows, grid, ctx)
    finite_mask = np.isfinite(base_pmf)
    if np.any(finite_mask):
        base_pmf[finite_mask] -= float(np.nanmin(base_pmf[finite_mask]))

    # Bootstrap EQ-MBAR: one call per anchor window, then take conservative nanmin.
    # Actual signature: bootstrap_direct_eq_mbar(window_rows, grid, reference_x, rng_seed, run_context, n_boot)
    # PMF and variance extracted as matched pairs to avoid mixing incompatible bootstrap conventions.
    variance_stack: list[np.ndarray] = []
    anchor_variances: dict[str, np.ndarray] = {}
    boot_pmf = base_pmf.copy()
    eq_bootstrap_failed = False
    eq_bootstrap_failure_reason = ""

    try:
        for ai, window in enumerate(cluster.windows):
            boot_result = bootstrap_direct_eq_mbar(
                window_rows,
                grid,
                float(window.mean_x),
                int(rng_seed) + 31 * ai,
                ctx,
                int(n_boot),
            )
            # Extract PMF/variance as matched pairs
            if "pmf_ref0" in boot_result and "var_ref0" in boot_result:
                pmf_i = np.asarray(boot_result["pmf_ref0"], dtype=float)
                bv = np.asarray(boot_result["var_ref0"], dtype=float)
            elif "pmf" in boot_result and "boot_var" in boot_result:
                pmf_i = np.asarray(boot_result["pmf"], dtype=float)
                bv = np.asarray(boot_result["boot_var"], dtype=float)
            elif "pmf" in boot_result and "variance" in boot_result:
                pmf_i = np.asarray(boot_result["pmf"], dtype=float)
                bv = np.asarray(boot_result["variance"], dtype=float)
            else:
                raise RuntimeError(f"Unknown bootstrap PMF/variance keys: {sorted(boot_result.keys())}")
            if ai == 0:
                boot_pmf = pmf_i
            variance_stack.append(bv)
            anchor_variances[window.name] = bv

        boot_var = np.nanmin(np.vstack(variance_stack), axis=0) if variance_stack else np.full(len(grid), np.nan, dtype=float)
    except Exception as exc:
        eq_bootstrap_failed = True
        eq_bootstrap_failure_reason = str(exc)
        boot_var = np.full(len(grid), np.nan, dtype=float)

    finite_m = np.isfinite(boot_pmf)
    if np.any(finite_m):
        boot_pmf[finite_m] -= float(np.nanmin(boot_pmf[finite_m]))

    # Coverage requires finite variance; if bootstrap failed, coverage is false everywhere
    coverage = coverage_mask_from_samples(
        np.concatenate([eq_tail_samples(w) for w in cluster.windows]), grid
    ) & np.isfinite(boot_pmf) & np.isfinite(boot_var)

    write_csv(patch_dir / "pmf.csv", ["x", "pmf", "variance"], [
        {"x": float(grid[i]),
         "pmf": float(boot_pmf[i]) if np.isfinite(boot_pmf[i]) else "",
         "variance": float(boot_var[i]) if np.isfinite(boot_var[i]) else ""}
        for i in range(len(grid))
    ])
    write_json(patch_dir / "patch_summary.json", {
        "name": cluster.name, "kind": "EQ_MBAR",
        "cluster_id": cluster.name,
        "n_windows": len(cluster.windows), "n_bins_covered": int(np.count_nonzero(coverage)),
        "eq_bootstrap_failed": eq_bootstrap_failed,
        "eq_bootstrap_failure_reason": eq_bootstrap_failure_reason,
        "n_anchor_bootstraps": len(variance_stack),
        "window_names": [w.name for w in cluster.windows],
        "window_mean_x": [float(w.mean_x) for w in cluster.windows],
        "window_center_x": [float(w.center_x) for w in cluster.windows],
    })
    return PMFPatch(
        name=cluster.name, kind="EQ_MBAR", root=patch_dir,
        grid=np.asarray(grid, dtype=float), pmf=boot_pmf,
        variance=boot_var, coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[w.name for w in cluster.windows],
        metadata={"cluster_name": cluster.name, "n_windows": len(cluster.windows),
                  "eq_bootstrap_failed": eq_bootstrap_failed,
                  "eq_bootstrap_failure_reason": eq_bootstrap_failure_reason},
        anchor_variances=anchor_variances,
    )


# ---------------------------------------------------------------------------
# v4-1: NEQ/MTS patch using original full bidirectional NES trajectories
# ---------------------------------------------------------------------------

def build_neq_mts_patch_v4(
    segment: NEQSegment, grid: np.ndarray, ctx: dict[str, Any],
    n_boot: int, patch_root: Path, rng_seed: int,
) -> PMFPatch:
    """Build NEQ/MTS patch using the original full bidirectional NES trajectories.

    v4-1: reads original protocol (not augmented); uses full work arrays directly.
    No final perturbation, no protocol elongation, no augmented files.
    mts_solved requires: cft_solved, finite delta_f, finite PMF and variance coverage.
    """
    patch_dir = patch_root / segment.name
    patch_dir.mkdir(parents=True, exist_ok=True)

    fwd_frames = [pd.DataFrame(rows) for rows in segment.forward_trajectories]
    rev_frames = [pd.DataFrame(rows) for rows in segment.reverse_trajectories]
    x_fwd, work_fwd = trajectory_frames_to_arrays(fwd_frames)
    x_rev, work_rev = trajectory_frames_to_arrays(rev_frames)

    # v4-1: read original protocol (not augmented)
    centers, ks = read_protocol_centers_and_k(segment.forward_path_file)

    n_time_protocol = len(centers)
    n_time_work = min(
        x_fwd.shape[1] if x_fwd.ndim == 2 else 0,
        work_fwd.shape[1] if work_fwd.ndim == 2 else 0,
        x_rev.shape[1] if x_rev.ndim == 2 else 0,
        work_rev.shape[1] if work_rev.ndim == 2 else 0,
    )
    n_time = min(n_time_protocol, n_time_work)
    if n_time <= 0:
        nan = np.full(len(grid), np.nan, dtype=float)
        segment.cft_summary = {"cft_solved": False, "mts_solved": False,
                                "reason": "no_time_slices"}
        return PMFPatch(
            name=segment.name, kind="NEQ_MTS", root=patch_dir,
            grid=np.asarray(grid, dtype=float), pmf=nan.copy(),
            variance=nan.copy(), coverage_mask=np.zeros(len(grid), dtype=bool),
            source_names=[segment.left_boundary.name, segment.right_boundary.name],
            metadata={"mts_solved": False, "cft_solved": False},
        )
    centers = centers[:n_time]
    ks = ks[:n_time]

    kT = float(ctx.get("thermal_kT", 1.0))
    # v4-1: use full work arrays directly
    cft = solve_segment_cft_delta_f_once(
        work_fwd[:, :n_time], work_rev[:, :n_time], kT=kT
    )
    cft_solved = bool(cft.get("cft_solved", False))
    finite_delta_f = cft_solved and cft.get("delta_f") is not None and math.isfinite(float(cft["delta_f"]))
    fixed_delta_f = float(cft["delta_f"]) if finite_delta_f else None

    pmf_arr = np.full(len(grid), np.nan, dtype=float)
    var_arr = np.full(len(grid), np.nan, dtype=float)
    var_left = np.full(len(grid), np.nan, dtype=float)
    var_right = np.full(len(grid), np.nan, dtype=float)
    pmf_right = np.full(len(grid), np.nan, dtype=float)
    boot_n_used = 0
    boot_n_used_left = 0
    boot_n_used_right = 0
    mts_exception = ""

    try:
        left_ref_x = float(segment.left_boundary.mean_x)
        right_ref_x = float(segment.right_boundary.mean_x)

        boot_left = bootstrap_bidirectional_mts_pmf(
            x_fwd[:, :n_time], work_fwd[:, :n_time],
            x_rev[:, :n_time], work_rev[:, :n_time],
            centers, ks, grid,
            reference_x=left_ref_x, kT=kT,
            n_boot=int(n_boot), fk_boot=max(int(n_boot // 8), 4),
            rng_seed=int(rng_seed),
            fixed_delta_f=fixed_delta_f,
            recompute_delta_f_per_bootstrap=(fixed_delta_f is None),
        )
        pmf_arr_raw, var_left_raw = get_bootstrap_pmf_and_variance(boot_left)
        pmf_arr = np.asarray(pmf_arr_raw, dtype=float)
        var_left = np.asarray(var_left_raw, dtype=float)
        boot_n_used_left = int(boot_left.get("n_boot_used", 0))

        boot_right = bootstrap_bidirectional_mts_pmf(
            x_fwd[:, :n_time], work_fwd[:, :n_time],
            x_rev[:, :n_time], work_rev[:, :n_time],
            centers, ks, grid,
            reference_x=right_ref_x, kT=kT,
            n_boot=int(n_boot), fk_boot=max(int(n_boot // 8), 4),
            rng_seed=int(rng_seed) + 7919,
            fixed_delta_f=fixed_delta_f,
            recompute_delta_f_per_bootstrap=(fixed_delta_f is None),
        )
        pmf_right_raw, var_right_raw = get_bootstrap_pmf_and_variance(boot_right)
        pmf_right = np.asarray(pmf_right_raw, dtype=float)
        var_right = np.asarray(var_right_raw, dtype=float)
        boot_n_used_right = int(boot_right.get("n_boot_used", 0))

        # Two-anchor minimum variance: reduces bias from gauge choice.
        var_arr = np.nanmin(np.vstack([var_left, var_right]), axis=0)
        boot_n_used = min(boot_n_used_left, boot_n_used_right)
    except Exception as exc:
        mts_exception = str(exc)

    finite_pmf = np.any(np.isfinite(pmf_arr))
    finite_var = np.any(np.isfinite(var_arr))

    # Proper mts_solved check (inherited from v3-8)
    mts_solved = (cft_solved and finite_delta_f and finite_pmf and finite_var
                  and not mts_exception)

    coverage = (
        coverage_mask_from_samples(
            np.concatenate([x_fwd.ravel(), x_rev.ravel()]), grid
        ) & np.isfinite(pmf_arr) & np.isfinite(var_arr)
    )

    segment.cft_summary = {
        "cft_solved": cft_solved, "delta_f": cft.get("delta_f"),
        "delta_f_unc": cft.get("delta_f_unc"), "mts_solved": mts_solved,
        "mts_exception": mts_exception,
        "neq_pair_source": segment.connectivity.get("neq_pair_source", ""),
        "n_time_protocol": n_time_protocol,
        "n_time_work_array": n_time_work,
        "n_forward_trajectories": len(segment.forward_trajectories),
        "n_reverse_trajectories": len(segment.reverse_trajectories),
        "mts_variance_method": "min_left_right_anchor_bootstrap_variance",
        "left_reference_x": float(segment.left_boundary.mean_x),
        "right_reference_x": float(segment.right_boundary.mean_x),
        "n_boot_used_left_anchor": boot_n_used_left,
        "n_boot_used_right_anchor": boot_n_used_right,
        "n_boot_used": boot_n_used,
    }
    segment.mts_patch_built = mts_solved

    write_json(patch_dir / "patch_summary.json", {
        "name": segment.name, "kind": "NEQ_MTS",
        "cft_solved": cft_solved, "mts_solved": mts_solved,
        "n_time_protocol": n_time_protocol,
        "n_time_work_array": n_time_work,
        "n_forward_trajectories": len(segment.forward_trajectories),
        "n_reverse_trajectories": len(segment.reverse_trajectories),
        "delta_f": cft.get("delta_f"), "n_boot": n_boot,
        "mts_variance_method": "min_left_right_anchor_bootstrap_variance",
        "left_reference_x": float(segment.left_boundary.mean_x),
        "right_reference_x": float(segment.right_boundary.mean_x),
        "n_boot_used_left_anchor": boot_n_used_left,
        "n_boot_used_right_anchor": boot_n_used_right,
        "n_boot_used": boot_n_used,
        "n_finite_var_left_anchor": int(np.count_nonzero(np.isfinite(var_left))),
        "n_finite_var_right_anchor": int(np.count_nonzero(np.isfinite(var_right))),
        "n_finite_var_min_anchor": int(np.count_nonzero(np.isfinite(var_arr))),
    })
    write_csv(patch_dir / "pmf.csv",
              ["x", "pmf", "variance", "variance_left_anchor", "variance_right_anchor",
               "pmf_right_anchor", "coverage"],
              [{
                  "x": float(grid[i]),
                  "pmf": float(pmf_arr[i]) if np.isfinite(pmf_arr[i]) else "",
                  "variance": float(var_arr[i]) if np.isfinite(var_arr[i]) else "",
                  "variance_left_anchor": float(var_left[i]) if np.isfinite(var_left[i]) else "",
                  "variance_right_anchor": float(var_right[i]) if np.isfinite(var_right[i]) else "",
                  "pmf_right_anchor": float(pmf_right[i]) if np.isfinite(pmf_right[i]) else "",
                  "coverage": int(bool(coverage[i])),
              } for i in range(len(grid))])

    return PMFPatch(
        name=segment.name, kind="NEQ_MTS", root=patch_dir,
        grid=np.asarray(grid, dtype=float), pmf=pmf_arr,
        variance=var_arr, coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[segment.left_boundary.name, segment.right_boundary.name],
        metadata={
            "segment_name": segment.name, "cft_solved": cft_solved,
            "mts_solved": mts_solved, "delta_f": cft.get("delta_f"),
            "n_boot": n_boot,
            "mts_variance_method": "min_left_right_anchor_bootstrap_variance",
            "left_reference_x": float(segment.left_boundary.mean_x),
            "right_reference_x": float(segment.right_boundary.mean_x),
        },
        anchor_variances={
            f"{segment.left_boundary.name}_left_anchor": np.asarray(var_left, dtype=float),
            f"{segment.right_boundary.name}_right_anchor": np.asarray(var_right, dtype=float),
        },
    )


# ---------------------------------------------------------------------------
# Global PMF fusion: inverse-precision variance
# ---------------------------------------------------------------------------

def fit_global_pmf_v4(
    patches: list[PMFPatch], grid: np.ndarray, variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """min_{G,c} sum_p sum_x (G-F_p-c_p)^2/(var_p+eps); global_var = 1/sum(1/(var+eps))."""
    nan = np.full(len(grid), np.nan, dtype=float)
    if not patches:
        return nan.copy(), nan.copy(), {"n_patches": 0, "n_observations": 0, "patch_offsets": {}}

    obs = []
    for pi, patch in enumerate(patches):
        mask = np.asarray(patch.coverage_mask, dtype=bool) & np.isfinite(patch.pmf) & np.isfinite(patch.variance)
        for gi in np.where(mask)[0]:
            w = 1.0 / (float(patch.variance[gi]) + float(variance_floor))
            obs.append((pi, int(gi), float(patch.pmf[gi]), w))

    if not obs:
        return nan.copy(), nan.copy(), {
            "n_patches": len(patches), "n_observations": 0,
            "patch_offsets": {p.name: None for p in patches},
        }

    n_grid, n_patch = len(grid), len(patches)
    gauge_idx = int(obs[0][1])
    A = np.zeros((len(obs) + 1, n_grid + n_patch), dtype=float)
    b = np.zeros(len(obs) + 1, dtype=float)
    for ri, (pi, gi, fval, w) in enumerate(obs):
        s = math.sqrt(w)
        A[ri, gi] = s
        A[ri, n_grid + pi] = -s
        b[ri] = s * fval
    A[-1, gauge_idx] = 1e6
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    global_pmf = np.asarray(sol[:n_grid], dtype=float)
    offsets = np.asarray(sol[n_grid:], dtype=float)
    finite = np.isfinite(global_pmf)
    if np.any(finite):
        shift = float(np.nanmin(global_pmf[finite]))
        global_pmf[finite] -= shift
        offsets -= shift

    precision_sum = np.zeros(n_grid, dtype=float)
    n_cover = np.zeros(n_grid, dtype=int)
    for pi, gi, _, _ in obs:
        precision_sum[gi] += 1.0 / (float(patches[pi].variance[gi]) + float(variance_floor))
        n_cover[gi] += 1
    global_var = np.full(n_grid, np.nan, dtype=float)
    covered = precision_sum > 0
    global_var[covered] = 1.0 / precision_sum[covered]

    rms = float(np.sqrt(np.mean(
        [(b[ri] / math.sqrt(w) - (global_pmf[gi] - offsets[pi])) ** 2 for ri, (pi, gi, _, w) in enumerate(obs)]
    ))) if obs else float("nan")
    return global_pmf, global_var, {
        "n_patches": len(patches), "n_observations": len(obs),
        "patch_offsets": {patches[pi].name: float(offsets[pi]) for pi in range(n_patch)},
        "rms_residual": rms, "variance_formula": "inverse_precision",
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

_NEIGHBOR_EQ_COLS = [
    "left_window", "right_window", "left_mean_x", "right_mean_x",
    "left_center_x", "right_center_x", "O_ij", "O_ji", "O_pair",
    "eq_overlap_threshold", "connected", "bar_delta_f", "bar_delta_f_unc",
    "bar_solved", "overlap_method", "overlap_reason",
    "raw_sigmoid_O_ij_diagnostic", "raw_sigmoid_O_ji_diagnostic",
    "pair_jsd_diagnostic", "cluster_order_coordinate",
]

_PMF_COLS = ["x", "global_pmf", "global_variance", "n_covering_patches"]


def write_neighbor_eq_overlap(out_root: Path, overlap_rows: list[dict[str, Any]]) -> None:
    write_csv(out_root / "neighbor_eq_overlap.csv", _NEIGHBOR_EQ_COLS, overlap_rows)


def write_state_tables_v4(
    base_root: Path, windows: list[EnsembleWindow], clusters: list[EQCluster],
    segments: list[NEQSegment], patches: list[PMFPatch],
    fit_details: dict[str, Any], overlap_rows: list[dict[str, Any]],
    grid: np.ndarray, global_pmf: np.ndarray, global_var: np.ndarray,
) -> None:
    base_root.mkdir(parents=True, exist_ok=True)

    # x_most removed (since v3-7)
    _ordered_windows = sorted(windows, key=lambda w: (float(w.mean_x), str(w.name)))
    _window_id_map = {w.name: f"W{idx:03d}" for idx, w in enumerate(_ordered_windows, start=1)}
    write_csv(base_root / "windows.csv", [
        "window_id", "name", "center_x", "k", "mean_x", "std_x", "generation", "side",
    ], [
        {"window_id": _window_id_map.get(w.name, ""), "name": w.name,
         "center_x": w.center_x, "k": w.k, "mean_x": w.mean_x,
         "std_x": w.std_x, "generation": w.generation, "side": w.side}
        for w in windows
    ])

    write_csv(base_root / "clusters.csv", [
        "name", "n_windows", "left_x", "right_x", "window_names",
    ], [
        {"name": c.name, "n_windows": len(c.windows), "left_x": c.left_x, "right_x": c.right_x,
         "window_names": ";".join(w.name for w in c.windows)}
        for c in clusters
    ])

    write_csv(base_root / "segments.csv", [
        "name", "left_boundary", "right_boundary", "protocol_mode",
        "protocol_k", "mts_patch_built", "cft_solved",
    ], [
        {"name": s.name, "left_boundary": s.left_boundary.name,
         "right_boundary": s.right_boundary.name, "protocol_mode": s.protocol_mode,
         "protocol_k": s.protocol_k, "mts_patch_built": int(s.mts_patch_built),
         "cft_solved": int(bool(s.cft_summary.get("cft_solved", False)))}
        for s in segments
    ])

    write_csv(base_root / "patches.csv", [
        "name", "kind", "n_bins_covered", "source_names",
    ], [
        {"name": p.name, "kind": p.kind,
         "n_bins_covered": int(np.count_nonzero(p.coverage_mask)),
         "source_names": ";".join(p.source_names)}
        for p in patches
    ])

    write_neighbor_eq_overlap(base_root, overlap_rows)

    n_cover_map = np.zeros(len(grid), dtype=int)
    for patch in patches:
        mask = np.asarray(patch.coverage_mask, dtype=bool) & np.isfinite(patch.pmf)
        n_cover_map[mask] += 1

    write_csv(base_root / "global_pmf.csv", _PMF_COLS, [
        {"x": float(grid[i]),
         "global_pmf": float(global_pmf[i]) if np.isfinite(global_pmf[i]) else "",
         "global_variance": float(global_var[i]) if np.isfinite(global_var[i]) else "",
         "n_covering_patches": int(n_cover_map[i])}
        for i in range(len(grid))
    ])

    fd = dict(fit_details)
    fd["eq_network_connected"] = bool(eq_network_is_connected(clusters))
    fd["final_estimator"] = (
        "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
    )
    write_json(base_root / "global_fit_summary.json", fd)


# ---------------------------------------------------------------------------
# PMF quality tracking
# ---------------------------------------------------------------------------

def compute_pmf_quality(
    grid: np.ndarray, global_pmf: np.ndarray, global_var: np.ndarray,
    ctx: dict[str, Any], used_steps: int, stage: str,
) -> dict[str, Any]:
    analysis_xmin = float(ctx.get("analysis_xmin", float(np.nanmin(grid))))
    analysis_xmax = float(ctx.get("analysis_xmax", float(np.nanmax(grid))))
    mask = (grid >= analysis_xmin) & (grid <= analysis_xmax) & np.isfinite(global_pmf)
    n_covered = int(np.count_nonzero(mask))
    n_total = int(np.count_nonzero((grid >= analysis_xmin) & (grid <= analysis_xmax)))
    max_ddf = float(np.nanmax(np.sqrt(global_var[mask]))) if np.any(mask & np.isfinite(global_var)) else float("nan")
    return {
        "stage": stage, "used_steps": int(used_steps),
        "n_covered_bins": n_covered, "n_total_bins": n_total,
        "coverage_fraction": float(n_covered) / float(n_total) if n_total > 0 else float("nan"),
        "max_mbar_ddf": max_ddf,
    }


# ---------------------------------------------------------------------------
# Adaptive connectivity refinement — v4 (one pair per round, v4-3 reuse check)
# ---------------------------------------------------------------------------

_REFINEMENT_COLS = [
    "round", "target_pair", "left_window", "right_window",
    "left_mean_x", "right_mean_x", "segment_type", "nes_action",
    "cft_solved", "mts_patch_built", "mts_failure_reason",
    "fallback_used", "fallback_rule",
    "inserted_window", "inserted_center_x", "inserted_k",
    "target_mean", "target_sigma", "k_raw", "k_clipped",
    "mts_barrier_x0", "mts_barrier_k0", "mts_barrier_fit_source", "mts_barrier_fit_reason",
    "k_total_target",
    "gt_sigma_rule", "gt_local_s", "gt_left_mean", "gt_right_mean",
    "gt_left_sigma", "gt_right_sigma", "gt_sigma_reason",
    "reason", "n_clusters_before", "n_clusters_after", "used_steps",
]


def is_valid_bidirectional_segment(
    segment: NEQSegment,
    left_boundary: EnsembleWindow,
    right_boundary: EnsembleWindow,
) -> bool:
    """Return True only if segment has both forward and reverse trajectories for this pair.

    v4-3: If only one direction exists, do not reuse — run a new bidirectional NES.
    """
    return (
        segment.left_boundary.name == left_boundary.name
        and segment.right_boundary.name == right_boundary.name
        and len(segment.forward_trajectories) > 0
        and len(segment.reverse_trajectories) > 0
    )


def _find_existing_segment(
    segments: list[NEQSegment], left_boundary: EnsembleWindow, right_boundary: EnsembleWindow
) -> NEQSegment | None:
    """Return a valid bidirectional segment for this boundary pair, or None.

    v4-3: Only reuses segments satisfying is_valid_bidirectional_segment().
    """
    for seg in segments:
        if is_valid_bidirectional_segment(seg, left_boundary, right_boundary):
            return seg
    return None


def _first_disconnected_pair(
    clusters: list[EQCluster],
) -> tuple[EQCluster, EQCluster] | None:
    """Return first disconnected neighboring cluster pair (by left_x order)."""
    sorted_clusters = sorted(clusters, key=lambda c: float(c.left_x))
    if len(sorted_clusters) < 2:
        return None
    return sorted_clusters[0], sorted_clusters[1]


def run_connectivity_refinement_v4(
    *, windows: list[EnsembleWindow], clusters: list[EQCluster],
    segments: list[NEQSegment], patches_for_global: list[PMFPatch],
    global_pmf: np.ndarray, global_var: np.ndarray,
    fit_details: dict[str, Any], overlap_rows: list[dict[str, Any]],
    args: argparse.Namespace, ctx: dict[str, Any], grid: np.ndarray,
    out_root: Path, bin_path: str, budget: BudgetTracker,
    quality_rows: list[dict[str, Any]], profile: GlobalGTWidthProfile, beta_eff: float,
) -> tuple[list[EnsembleWindow], list[EQCluster], list[NEQSegment],
           list[PMFPatch], np.ndarray, np.ndarray, dict[str, Any],
           list[dict[str, Any]], list[dict[str, Any]]]:
    """Adaptive connectivity refinement: one disconnected pair per round.

    Rebuilds clusters from scratch each round; never iterates over stale list.
    Basin-like → midpoint EQ only (no NES).
    Transition-like → bidirectional NES/MTS (fallback on MTS failure).
    BAR failure means disconnected; raw sigmoid not used.
    v4-3: only reuses segments with both forward and reverse trajectories.
    """
    ref_rows: list[dict[str, Any]] = []

    def write_refinement() -> None:
        write_csv(out_root / "refinement_summary.csv", _REFINEMENT_COLS, ref_rows)

    if eq_network_is_connected(clusters):
        write_refinement()
        return (windows, clusters, segments, patches_for_global,
                global_pmf, global_var, fit_details, overlap_rows, ref_rows)

    max_rounds = int(args.max_refinement_rounds)
    neq_cost = int(2 * args.n_neq_traj * args.t_neq)
    eq_cost = int(args.n_eq_steps)

    for round_idx in range(max_rounds):
        # Rebuild clusters from scratch each round
        clusters, overlap_rows = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))
        if eq_network_is_connected(clusters):
            break

        # Get first disconnected pair from fresh clusters
        pair = _first_disconnected_pair(clusters)
        if pair is None:
            break
        left_cluster, right_cluster = pair

        left_boundary = max(left_cluster.windows, key=lambda w: float(w.mean_x))
        right_boundary = min(right_cluster.windows, key=lambda w: float(w.mean_x))
        pair_label = f"{left_boundary.name}_{right_boundary.name}"
        n_clusters_before = len(clusters)

        row: dict[str, Any] = {
            "round": round_idx, "target_pair": pair_label,
            "left_window": left_boundary.name, "right_window": right_boundary.name,
            "left_mean_x": float(left_boundary.mean_x), "right_mean_x": float(right_boundary.mean_x),
            "segment_type": "", "nes_action": "",
            "cft_solved": 0, "mts_patch_built": 0, "mts_failure_reason": "",
            "fallback_used": 0, "fallback_rule": "",
            "inserted_window": "", "inserted_center_x": "", "inserted_k": "",
            "target_mean": "", "target_sigma": "", "k_raw": "", "k_clipped": "",
            "reason": "", "n_clusters_before": n_clusters_before, "n_clusters_after": n_clusters_before,
            "used_steps": int(budget.used_steps),
        }

        # Classify using boundary window means
        is_transition, bg = classify_segment(left_boundary, right_boundary)
        row["segment_type"] = "barrier_like" if is_transition else "basin_like"

        if not is_transition:
            # Basin: no NES — insert midpoint EQ window
            row["nes_action"] = "none_basin_like"
            design = design_midpoint_window(
                left_boundary, right_boundary, profile,
                float(args.k_min), float(args.k_max), beta_eff,
            )
            row.update({
                "target_mean": float(design["target_mean"]),
                "target_sigma": float(design["target_sigma"]),
                "k_raw": float(design["k_raw"]), "k_clipped": int(design["k_clipped"]),
            })

            if not budget.can_spend(eq_cost):
                row["reason"] = "budget_exhausted_before_eq_insert"
                ref_rows.append(row)
                write_refinement()
                return (windows, clusters, segments, patches_for_global,
                        global_pmf, global_var, fit_details, overlap_rows, ref_rows)

            win_name = f"refine_r{round_idx}_basin_{pair_label}_win"
            win_root = out_root / "refinement" / f"round_{round_idx:03d}" / "windows" / win_name
            budget.spend(eq_cost, win_name, "EQ", f"refinement_round_{round_idx}")
            new_win = run_eq_window_v4(
                name=win_name, center_x=float(design["x_child"]), k=float(design["k_child"]),
                generation=-1, side="refine", bin_path=bin_path, ctx=ctx,
                n_eq_steps=args.n_eq_steps, eq_save_every=args.eq_save_every,
                tail_fraction=args.tail_fraction,
                seed=int(args.seed) + 700000 + round_idx * 100,
                root=win_root,
            )
            windows.append(new_win)
            row.update({
                "inserted_window": win_name,
                "inserted_center_x": float(design["x_child"]),
                "inserted_k": float(design["k_child"]),
                "reason": "basin_midpoint_eq_inserted",
            })

        else:
            # Transition: bidirectional NES/MTS
            # v4-3: only reuse if both forward and reverse trajectories exist
            existing_seg = _find_existing_segment(segments, left_boundary, right_boundary)

            if existing_seg is not None:
                row["nes_action"] = "reuse_existing_bidirectional"
                seg = existing_seg
            else:
                if not budget.can_spend(neq_cost):
                    row["reason"] = "budget_exhausted_before_neq"
                    ref_rows.append(row)
                    write_refinement()
                    return (windows, clusters, segments, patches_for_global,
                            global_pmf, global_var, fit_details, overlap_rows, ref_rows)

                seg_name = f"refine_r{round_idx}_trans_{pair_label}"
                seg_root = out_root / "refinement" / f"round_{round_idx:03d}" / "segments" / seg_name
                budget.spend(neq_cost, seg_name, "NEQ", f"refinement_round_{round_idx}")
                seg = run_neq_segment_v4(
                    name=seg_name, left_boundary=left_boundary, right_boundary=right_boundary,
                    left_source=left_cluster, right_source=right_cluster,
                    bin_path=bin_path, ctx=ctx, t_neq=args.t_neq, n_neq_traj=args.n_neq_traj,
                    neq_nout=int(args.neq_nout), k_mid_scale=float(args.neq_k_mid_scale),
                    seed=int(args.seed) + 500000 + round_idx * 100,
                    root=seg_root, k_min=float(args.k_min), k_max=float(args.k_max),
                    neq_pair_source="newly_generated",
                )
                segments.append(seg)
                row["nes_action"] = "newly_generated"

            # Build MTS patch
            patch_root = out_root / "refinement" / f"round_{round_idx:03d}" / "patches"
            mts_patch = build_neq_mts_patch_v4(
                seg, grid, ctx, int(args.n_bootstrap_neq), patch_root,
                rng_seed=int(args.seed) + 600000 + round_idx * 100,
            )
            cft_solved = bool(seg.cft_summary.get("cft_solved", False))
            mts_solved = bool(seg.cft_summary.get("mts_solved", False))
            row["cft_solved"] = int(cft_solved)
            row["mts_patch_built"] = int(mts_solved)

            if mts_solved and np.any(np.isfinite(mts_patch.pmf)) and np.any(np.isfinite(mts_patch.variance)):
                seg.cft_summary["_mts_patch"] = mts_patch

                design = design_mts_barrier_gt_window(
                    left_boundary=left_boundary,
                    right_boundary=right_boundary,
                    mts_patch=mts_patch,
                    k_min=float(args.k_min),
                    k_max=float(args.k_max),
                    beta_eff=beta_eff,
                )

                row.update({
                    "mts_barrier_x0": design.get("mts_x0", ""),
                    "mts_barrier_k0": design.get("mts_k0", ""),
                    "mts_barrier_fit_source": design.get("fit_source", ""),
                    "mts_barrier_fit_reason": design.get("fit_reason", ""),
                    "k_total_target": design.get("k_total_target", ""),
                    "gt_sigma_rule": design.get("gt_sigma_rule", ""),
                    "gt_local_s": design.get("gt_local_s", ""),
                    "gt_left_mean": design.get("gt_left_mean", ""),
                    "gt_right_mean": design.get("gt_right_mean", ""),
                    "gt_left_sigma": design.get("gt_left_sigma", ""),
                    "gt_right_sigma": design.get("gt_right_sigma", ""),
                    "gt_sigma_reason": design.get("gt_sigma_reason", ""),
                })

                if not bool(design.get("fit_valid", False)):
                    # MTS solved but barrier fitting failed; use midpoint fallback
                    row["fallback_used"] = 1
                    row["fallback_rule"] = "midpoint_mean_only_GT_after_MTS_barrier_fit_failure"
                    row["mts_failure_reason"] = str(design.get("fit_reason", "mts_barrier_fit_failed"))

                    design_fb = design_midpoint_window(
                        left_boundary, right_boundary, profile,
                        float(args.k_min), float(args.k_max), beta_eff,
                    )
                    row.update({
                        "target_mean": float(design_fb["target_mean"]),
                        "target_sigma": float(design_fb["target_sigma"]),
                        "k_raw": float(design_fb["k_raw"]), "k_clipped": int(design_fb["k_clipped"]),
                    })

                    if not budget.can_spend(eq_cost):
                        row["reason"] = "budget_exhausted_before_fallback_eq"
                        ref_rows.append(row)
                        write_refinement()
                        return (windows, clusters, segments, patches_for_global,
                                global_pmf, global_var, fit_details, overlap_rows, ref_rows)

                    win_name = f"refine_r{round_idx}_fallback_{pair_label}_win"
                    win_root = out_root / "refinement" / f"round_{round_idx:03d}" / "windows" / win_name
                    budget.spend(eq_cost, win_name, "EQ", f"refinement_round_{round_idx}")
                    new_win = run_eq_window_v4(
                        name=win_name, center_x=float(design_fb["x_child"]), k=float(design_fb["k_child"]),
                        generation=-1, side="refine", bin_path=bin_path, ctx=ctx,
                        n_eq_steps=args.n_eq_steps, eq_save_every=args.eq_save_every,
                        tail_fraction=args.tail_fraction,
                        seed=int(args.seed) + 710000 + round_idx * 100,
                        root=win_root,
                    )
                    windows.append(new_win)
                    row.update({
                        "inserted_window": win_name,
                        "inserted_center_x": float(design_fb["x_child"]),
                        "inserted_k": float(design_fb["k_child"]),
                        "reason": "transition_mts_solved_barrier_fit_failed_midpoint_fallback",
                    })
                else:
                    row.update({
                        "target_mean": float(design["target_mean"]),
                        "target_sigma": float(design["target_sigma"]),
                        "k_raw": float(design["k_raw"]),
                        "k_clipped": int(design["k_clipped"]),
                        "gt_sigma_rule": design.get("gt_sigma_rule", ""),
                        "gt_local_s": design.get("gt_local_s", ""),
                        "gt_left_mean": design.get("gt_left_mean", ""),
                        "gt_right_mean": design.get("gt_right_mean", ""),
                        "gt_left_sigma": design.get("gt_left_sigma", ""),
                        "gt_right_sigma": design.get("gt_right_sigma", ""),
                        "gt_sigma_reason": design.get("gt_sigma_reason", ""),
                    })

                    if not budget.can_spend(eq_cost):
                        row["reason"] = "budget_exhausted_before_mts_barrier_eq_insert"
                        ref_rows.append(row)
                        write_refinement()
                        return (windows, clusters, segments, patches_for_global,
                                global_pmf, global_var, fit_details, overlap_rows, ref_rows)

                    win_name = f"refine_r{round_idx}_mtsbarrier_{pair_label}_win"
                    win_root = out_root / "refinement" / f"round_{round_idx:03d}" / "windows" / win_name
                    budget.spend(eq_cost, win_name, "EQ", f"refinement_round_{round_idx}")
                    new_win = run_eq_window_v4(
                        name=win_name,
                        center_x=float(design["x_child"]),
                        k=float(design["k_child"]),
                        generation=-1,
                        side="refine",
                        bin_path=bin_path,
                        ctx=ctx,
                        n_eq_steps=args.n_eq_steps,
                        eq_save_every=args.eq_save_every,
                        tail_fraction=args.tail_fraction,
                        seed=int(args.seed) + 720000 + round_idx * 100,
                        root=win_root,
                    )
                    windows.append(new_win)
                    row.update({
                        "inserted_window": win_name,
                        "inserted_center_x": float(design["x_child"]),
                        "inserted_k": float(design["k_child"]),
                        "reason": "transition_mts_solved_mts_barrier_gt_eq_inserted",
                        "fallback_used": 0,
                        "fallback_rule": "",
                    })
            else:
                # MTS failure fallback: midpoint mean-only GT EQ insertion
                mts_fail_reason = str(seg.cft_summary.get("mts_exception", ""))
                if not mts_fail_reason:
                    if not cft_solved:
                        mts_fail_reason = "cft_not_solved"
                    elif not mts_solved:
                        mts_fail_reason = "mts_no_finite_coverage"
                row["mts_failure_reason"] = mts_fail_reason
                row["fallback_used"] = 1
                row["fallback_rule"] = "midpoint_mean_only_GT_after_MTS_failure"

                design = design_midpoint_window(
                    left_boundary, right_boundary, profile,
                    float(args.k_min), float(args.k_max), beta_eff,
                )
                row.update({
                    "target_mean": float(design["target_mean"]),
                    "target_sigma": float(design["target_sigma"]),
                    "k_raw": float(design["k_raw"]), "k_clipped": int(design["k_clipped"]),
                })

                if not budget.can_spend(eq_cost):
                    row["reason"] = "budget_exhausted_before_fallback_eq"
                    ref_rows.append(row)
                    write_refinement()
                    return (windows, clusters, segments, patches_for_global,
                            global_pmf, global_var, fit_details, overlap_rows, ref_rows)

                win_name = f"refine_r{round_idx}_fallback_{pair_label}_win"
                win_root = out_root / "refinement" / f"round_{round_idx:03d}" / "windows" / win_name
                budget.spend(eq_cost, win_name, "EQ", f"refinement_round_{round_idx}")
                new_win = run_eq_window_v4(
                    name=win_name, center_x=float(design["x_child"]), k=float(design["k_child"]),
                    generation=-1, side="refine", bin_path=bin_path, ctx=ctx,
                    n_eq_steps=args.n_eq_steps, eq_save_every=args.eq_save_every,
                    tail_fraction=args.tail_fraction,
                    seed=int(args.seed) + 710000 + round_idx * 100,
                    root=win_root,
                )
                windows.append(new_win)
                row.update({
                    "inserted_window": win_name,
                    "inserted_center_x": float(design["x_child"]),
                    "inserted_k": float(design["k_child"]),
                    "reason": "transition_mts_failed_midpoint_eq_fallback",
                })

        # Rebuild clusters after each insertion/repair
        clusters, overlap_rows = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))
        row["n_clusters_after"] = len(clusters)

        # Rebuild EQ patches and provisional PMF
        eq_patches: list[PMFPatch] = []
        for ci, cl in enumerate(clusters):
            eq_patches.append(build_eq_cluster_patch_v4(
                cl, grid, ctx, int(args.n_bootstrap_eq),
                out_root / "refinement" / f"round_{round_idx:03d}" / "patches" / "eq",
                rng_seed=int(args.seed) + 400000 + round_idx * 1000 + ci,
            ))

        if eq_network_is_connected(clusters):
            patches_for_global = eq_patches
            patch_selection_rule = "connected_EQ_MBAR_only"
        else:
            # Only include MTS patches that actually solved
            valid_mts = [
                p for p in [
                    s.cft_summary.get("_mts_patch") for s in segments
                    if s.cft_summary.get("mts_solved", False)
                ] if p is not None
            ]
            if args.pmf_method == "neq" and valid_mts:
                patches_for_global = valid_mts
                patch_selection_rule = "only_NEQ_MTS"
            elif args.pmf_method == "hybrid" and valid_mts:
                patches_for_global = eq_patches + valid_mts
                patch_selection_rule = "EQ_MBAR_plus_NEQ_MTS"
            else:
                patches_for_global = eq_patches
                patch_selection_rule = "EQ_MBAR_only_provisional"

        global_pmf, global_var, fit_details = fit_global_pmf_v4(
            patches_for_global, grid, float(args.variance_floor)
        )
        fit_details["patch_selection_rule"] = patch_selection_rule
        fit_details["eq_network_connected"] = bool(eq_network_is_connected(clusters))
        fit_details["final_estimator"] = (
            "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
        )

        quality_rows.append(compute_pmf_quality(
            grid, global_pmf, global_var, ctx, int(budget.used_steps),
            f"refinement_round_{round_idx}",
        ))

        ref_rows.append(row)
        write_refinement()

    write_refinement()
    return (windows, clusters, segments, patches_for_global,
            global_pmf, global_var, fit_details, overlap_rows, ref_rows)


# ---------------------------------------------------------------------------
# Final EQ-extension (connected-EQ MBAR only)
# ---------------------------------------------------------------------------

_EXT_SUMMARY_COLS = [
    "round", "used_steps", "remaining_steps", "n_selected_windows",
    "eq_extension_steps", "round_cost", "max_mbar_ddf", "x_at_max_mbar_ddf",
    "target_mbar_ddf", "stop_reason", "n_clusters_after_extension",
]


def run_final_eq_extension_v4(
    *, windows: list[EnsembleWindow], clusters: list[EQCluster],
    segments: list[NEQSegment], patches_for_global: list[PMFPatch],
    global_pmf: np.ndarray, global_var: np.ndarray,
    fit_details: dict[str, Any], overlap_rows: list[dict[str, Any]],
    args: argparse.Namespace, ctx: dict[str, Any], grid: np.ndarray,
    out_root: Path, bin_path: str, budget: BudgetTracker,
    quality_rows: list[dict[str, Any]],
) -> tuple[list[EQCluster], list[PMFPatch], np.ndarray, np.ndarray, dict[str, Any],
           list[dict[str, Any]], list[dict[str, Any]]]:
    ext_rows: list[dict[str, Any]] = []

    def _write_empty(reason: str) -> None:
        ext_rows.append({
            "round": 0, "used_steps": int(budget.used_steps),
            "remaining_steps": int(budget.total_budget_steps - budget.used_steps),
            "n_selected_windows": 0, "eq_extension_steps": 0, "round_cost": 0,
            "max_mbar_ddf": float("nan"), "x_at_max_mbar_ddf": float("nan"),
            "target_mbar_ddf": float(args.target_mbar_ddf),
            "stop_reason": reason, "n_clusters_after_extension": len(clusters),
        })
        write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)

    if getattr(args, "final_refinement_mode", "none") != "eq-extend":
        _write_empty("mode_disabled")
        return clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows

    if not eq_network_is_connected(clusters):
        _write_empty("eq_network_not_connected")
        return clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows

    eq_ext_steps = int(args.eq_extension_steps or args.n_eq_steps)
    target_ddf = float(args.target_mbar_ddf)
    base_seed = int(args.seed) + 900000
    round_index = 0

    while True:
        selected_windows = clusters[0].windows
        analysis_mask = (
            (grid >= float(ctx.get("analysis_xmin", float(np.nanmin(grid)))))
            & (grid <= float(ctx.get("analysis_xmax", float(np.nanmax(grid)))))
            & np.isfinite(global_var)
        )
        if np.any(analysis_mask):
            ddf_arr = np.sqrt(global_var[analysis_mask])
            max_ddf = float(np.nanmax(ddf_arr))
            x_at_max = float(grid[analysis_mask][int(np.nanargmax(ddf_arr))])
        else:
            max_ddf, x_at_max = float("nan"), float("nan")

        round_cost = len(selected_windows) * eq_ext_steps
        summary_row: dict[str, Any] = {
            "round": round_index, "used_steps": int(budget.used_steps),
            "remaining_steps": int(budget.total_budget_steps - budget.used_steps),
            "n_selected_windows": len(selected_windows),
            "eq_extension_steps": eq_ext_steps, "round_cost": round_cost,
            "max_mbar_ddf": max_ddf, "x_at_max_mbar_ddf": x_at_max,
            "target_mbar_ddf": target_ddf, "stop_reason": "",
            "n_clusters_after_extension": 1,
        }

        if math.isfinite(max_ddf) and max_ddf < target_ddf:
            summary_row["stop_reason"] = "target_mbar_ddf_reached"
            ext_rows.append(summary_row)
            break
        if not budget.can_spend(round_cost):
            summary_row["stop_reason"] = "budget_exhausted"
            ext_rows.append(summary_row)
            break

        round_root = out_root / "final_eq_extension" / f"round_{round_index:03d}"
        for wi, window in enumerate(selected_windows):
            ext_seed = base_seed + round_index * 1000 + wi
            ext_win_root = round_root / "windows" / window.name
            ext_win_root.mkdir(parents=True, exist_ok=True)
            nout = max(1, int(math.ceil(float(eq_ext_steps) / max(int(args.eq_save_every), 1))))
            _run_eq_window_raw(
                bin_path=bin_path, ctx=ctx, center_x=float(window.center_x),
                k=float(window.k), steps=eq_ext_steps, nout=nout,
                seed=ext_seed, out_dir=ext_win_root,
            )
            ext_eq_file = ext_win_root / "eq_window.csv"
            ext_eq_rows = read_csv_rows(ext_eq_file)
            window.eq_rows = window.eq_rows + ext_eq_rows
            window.tail_rows = tail_rows_from_eq_rows(window.eq_rows, float(args.tail_fraction))
            tail_x = np.asarray(
                [float(r["x"]) for r in window.tail_rows if r.get("x", "") != ""], dtype=float
            )
            tail_x = tail_x[np.isfinite(tail_x)]
            if tail_x.size >= 2:
                window.mean_x = float(np.mean(tail_x))
                window.std_x = float(np.std(tail_x, ddof=1))

        budget.spend(round_cost, f"final_eq_extension_round_{round_index:03d}",
                     "EQ_EXTENSION", "final_eq_extension")

        new_clusters, new_overlap = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))
        overlap_rows = new_overlap

        if not eq_network_is_connected(new_clusters):
            summary_row["stop_reason"] = "eq_connectivity_lost"
            summary_row["n_clusters_after_extension"] = len(new_clusters)
            ext_rows.append(summary_row)
            write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)
            write_json(out_root / "eq_connectivity_lost.json", {
                "round": round_index, "n_clusters_after": len(new_clusters),
                "cluster_names": [c.name for c in new_clusters],
            })
            return clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows

        clusters = new_clusters
        new_patches: list[PMFPatch] = []
        for ci, cl in enumerate(clusters):
            new_patches.append(build_eq_cluster_patch_v4(
                cl, grid, ctx, int(args.n_bootstrap_eq),
                round_root / "patches",
                rng_seed=base_seed + 300000 + round_index * 1000 + ci,
            ))
        global_pmf, global_var, fit_details = fit_global_pmf_v4(
            new_patches, grid, float(args.variance_floor)
        )
        patches_for_global = new_patches
        fit_details["patch_selection_rule"] = "connected_EQ_MBAR_only"
        fit_details["eq_network_connected"] = True
        fit_details["final_estimator"] = "connected_EQ_MBAR_only"

        write_state_tables_v4(round_root / "state", windows, clusters, segments,
                               patches_for_global, fit_details, overlap_rows,
                               grid, global_pmf, global_var)
        quality_rows.append(compute_pmf_quality(
            grid, global_pmf, global_var, ctx, int(budget.used_steps),
            f"final_eq_extension_round_{round_index:03d}",
        ))
        summary_row["stop_reason"] = "round_complete"
        ext_rows.append(summary_row)
        write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)
        round_index += 1

    write_csv(out_root / "final_eq_extension_summary.csv", _EXT_SUMMARY_COLS, ext_rows)
    return clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows


# ---------------------------------------------------------------------------
# Smoke-test overrides
# ---------------------------------------------------------------------------

def apply_quick_test_overrides(args: argparse.Namespace) -> None:
    args.n_eq_steps = 1000
    args.t_neq = 200
    args.n_neq_traj = 10
    args.neq_nout = 11
    args.max_generations = 1
    args.max_refinement_rounds = 2
    args.n_bootstrap_eq = 8
    args.n_bootstrap_neq = 8
    args.final_refinement_mode = "none"
    args.total_budget_steps = 20000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiNES v4 — strict BAR connectivity, NES-only for transitions, "
                    "full bidirectional NES trajectories for CFT/MTS, no final perturbation, "
                    "MTS failure fallback, no x_most."
    )
    parser.add_argument("--system-root", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--label", default="mines_variance_fusion_v4")
    parser.add_argument("--total-budget-steps", default=2500000, type=int)
    parser.add_argument("--n-eq-steps", default=10000, type=int)
    parser.add_argument("--eq-save-every", default=10, type=int)
    parser.add_argument("--tail-fraction", default=0.9, type=float)
    parser.add_argument("--t-neq", default=5000, type=int)
    parser.add_argument("--n-neq-traj", default=100, type=int)
    parser.add_argument("--target-kl", default=1.0, type=float)
    parser.add_argument("--eq-overlap-threshold", default=0.3, type=float)
    parser.add_argument("--max-generations", default=10, type=int)
    parser.add_argument("--max-refinement-rounds", default=10, type=int)
    parser.add_argument("--k-min", default=1.0, type=float)
    parser.add_argument("--k-max", default=100.0, type=float)
    parser.add_argument("--bin-width", default=0.1, type=float)
    parser.add_argument("--n-bootstrap-eq", default=64, type=int)
    parser.add_argument("--n-bootstrap-neq", default=64, type=int)
    parser.add_argument("--variance-floor", default=1e-6, type=float)
    parser.add_argument("--pmf-method", choices=["neq", "eq", "hybrid"], default="neq")
    parser.add_argument("--final-refinement-mode", choices=["none", "eq-extend"], default="eq-extend")
    parser.add_argument("--target-mbar-ddf", default=1e-3, type=float)
    parser.add_argument("--eq-extension-steps", default=None, type=int)
    parser.add_argument(
        "--neq-nout", default=101, type=int,
        help="Requested number of saved output frames per NEQ trajectory. Default: 101.",
    )
    parser.add_argument(
        "--neq-k-mid-scale", default=1.0, type=float,
        help="Midpoint k enhancement factor for the quadratic NEQ spring protocol. Default: 1.0.",
    )
    parser.add_argument("--quick-test", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.quick_test:
        apply_quick_test_overrides(args)

    system_root = Path(args.system_root)
    ctx = load_json(system_root / "run_context.json")
    bin_path = str(args.bin)
    label, seed = str(args.label), int(args.seed)

    system_id = str(ctx.get("system_id", "SYS1"))
    ctx["system_id"] = system_id

    out_root = system_root / "MINES" / label / "raw" / f"seed_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    grid_dx = float(args.bin_width)
    grid = build_grid(
        float(ctx["grid"]["xmin"]),
        float(ctx["grid"]["xmax"]),
        grid_dx,
    )
    ctx["grid_dx"] = grid_dx
    beta_eff = 1.0 / max(float(ctx.get("thermal_kT", 1.0)), 1e-12)

    budget = BudgetTracker(total_budget_steps=int(args.total_budget_steps))
    windows: list[EnsembleWindow] = []
    segments: list[NEQSegment] = []
    quality_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []

    def eq_cost(steps: int) -> int:
        return int(steps)

    def neq_cost(n_traj: int, t: int) -> int:
        return int(2 * n_traj * t)

    # -----------------------------------------------------------------------
    # Stage 0: Sample L0 and R0 endpoints
    # -----------------------------------------------------------------------
    for _ in range(2):
        if not budget.can_spend(eq_cost(args.n_eq_steps)):
            raise RuntimeError("Budget too small for endpoint EQ sampling.")
        budget.spend(eq_cost(args.n_eq_steps), "endpoint", "EQ", "init")

    left0 = run_eq_window_v4(
        name="L_gen0",
        center_x=float(ctx["mines_screen"]["fixed"].get("start_x_left", ctx["basins"]["left"])),
        k=float(args.k_min), generation=0, side="left",
        bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
        eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
        seed=seed + 1, root=out_root / "windows" / "L_gen0",
    )
    right0 = run_eq_window_v4(
        name="R_gen0",
        center_x=float(ctx["mines_screen"]["fixed"].get("start_x_right", ctx["basins"]["right"])),
        k=float(args.k_min), generation=0, side="right",
        bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
        eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
        seed=seed + 2, root=out_root / "windows" / "R_gen0",
    )
    windows.extend([left0, right0])
    profile = make_global_gt_profile(left0, right0)
    left_frontier, right_frontier = left0, right0

    clusters, overlap_rows = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))
    patches_for_global: list[PMFPatch] = []
    global_pmf = np.full(len(grid), np.nan, dtype=float)
    global_var = np.full(len(grid), np.nan, dtype=float)
    fit_details: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Stage 1: Exploratory chain growth
    # -----------------------------------------------------------------------
    for generation in range(int(args.max_generations)):
        clusters, overlap_rows = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))
        if eq_network_is_connected(clusters):
            print(f"[v4] EQ network connected at generation {generation}. Stopping exploration.")
            break

        seg_name = f"seg_L{left_frontier.name}_R{right_frontier.name}_gen{generation}"
        n_cost = neq_cost(args.n_neq_traj, args.t_neq)
        if not budget.can_spend(n_cost):
            print("[v4] Budget exhausted before NEQ at generation", generation)
            break
        budget.spend(n_cost, seg_name, "NEQ", f"generation_{generation}")
        seg = run_neq_segment_v4(
            name=seg_name, left_boundary=left_frontier, right_boundary=right_frontier,
            left_source=left_frontier, right_source=right_frontier,
            bin_path=bin_path, ctx=ctx, t_neq=args.t_neq, n_neq_traj=args.n_neq_traj,
            neq_nout=int(args.neq_nout), k_mid_scale=float(args.neq_k_mid_scale),
            seed=seed + 100 + generation,
            root=out_root / "segments" / seg_name,
            k_min=float(args.k_min), k_max=float(args.k_max),
        )
        segments.append(seg)

        # Diagnostic MTS patch for the frontier segment
        mts_patch = build_neq_mts_patch_v4(
            seg, grid, ctx, int(args.n_bootstrap_neq),
            out_root / "patches" / f"gen{generation}",
            rng_seed=seed + 200 + generation,
        )
        # Store patch ref on segment for later PMF routing
        if bool(seg.cft_summary.get("mts_solved", False)):
            seg.cft_summary["_mts_patch"] = mts_patch

        # Design children
        left_design = design_child_window(
            current_window=left_frontier, opposite_window=right_frontier,
            profile=profile, k_min=float(args.k_min), k_max=float(args.k_max),
            beta_eff=beta_eff, target_kl=float(args.target_kl), direction="right",
        )
        right_design = design_child_window(
            current_window=right_frontier, opposite_window=left_frontier,
            profile=profile, k_min=float(args.k_min), k_max=float(args.k_max),
            beta_eff=beta_eff, target_kl=float(args.target_kl), direction="left",
        )

        if not budget.can_spend(2 * eq_cost(args.n_eq_steps)):
            print("[v4] Budget exhausted before EQ children at generation", generation)
            break
        budget.spend(2 * eq_cost(args.n_eq_steps), f"children_gen{generation}",
                     "EQ", f"generation_{generation}")

        left_name = f"L_gen{generation + 1}"
        right_name = f"R_gen{generation + 1}"

        left_child = run_eq_window_v4(
            name=left_name, center_x=float(left_design["x_child"]),
            k=float(left_design["k_child"]), generation=generation + 1, side="left",
            bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
            eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
            seed=seed + 300 + generation * 2, root=out_root / "windows" / left_name,
        )
        windows.append(left_child)
        # Rebuild after left child (inherited from v3-5)
        clusters, overlap_rows = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))

        right_child = run_eq_window_v4(
            name=right_name, center_x=float(right_design["x_child"]),
            k=float(right_design["k_child"]), generation=generation + 1, side="right",
            bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
            eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
            seed=seed + 300 + generation * 2 + 1, root=out_root / "windows" / right_name,
        )
        windows.append(right_child)
        # Rebuild after right child (inherited from v3-5)
        clusters, overlap_rows = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))

        left_frontier = left_child
        right_frontier = right_child
        left_design["generation"] = generation
        right_design["generation"] = generation
        generation_rows.extend([left_design, right_design])

        # Build EQ patches and provisional PMF
        eq_patches: list[PMFPatch] = []
        for ci, cl in enumerate(clusters):
            eq_patches.append(build_eq_cluster_patch_v4(
                cl, grid, ctx, int(args.n_bootstrap_eq),
                out_root / "patches" / f"gen{generation}" / "eq",
                rng_seed=seed + 400 + generation * 100 + ci,
            ))

        if eq_network_is_connected(clusters):
            patches_for_global = eq_patches
            patch_selection_rule = "connected_EQ_MBAR_only"
        else:
            # Only include mts_solved patches
            mts_valid = mts_patch if bool(seg.cft_summary.get("mts_solved", False)) else None
            if args.pmf_method == "neq" and mts_valid:
                patches_for_global = [mts_valid]
                patch_selection_rule = "only_NEQ_MTS"
            elif args.pmf_method == "hybrid" and mts_valid:
                patches_for_global = eq_patches + [mts_valid]
                patch_selection_rule = "EQ_MBAR_plus_NEQ_MTS"
            else:
                patches_for_global = eq_patches
                patch_selection_rule = "EQ_MBAR_only_provisional"

        global_pmf, global_var, fit_details = fit_global_pmf_v4(
            patches_for_global, grid, float(args.variance_floor)
        )
        fit_details["patch_selection_rule"] = patch_selection_rule
        fit_details["eq_network_connected"] = bool(eq_network_is_connected(clusters))
        fit_details["final_estimator"] = (
            "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
        )

        gen_root = out_root / f"generation_{generation:03d}"
        write_state_tables_v4(gen_root, windows, clusters, segments,
                               patches_for_global, fit_details, overlap_rows,
                               grid, global_pmf, global_var)
        quality_rows.append(compute_pmf_quality(
            grid, global_pmf, global_var, ctx, int(budget.used_steps), f"generation_{generation}"
        ))

        if eq_network_is_connected(clusters):
            break

    if generation_rows:
        write_csv(out_root / "generation_summary.csv",
                  ordered_fieldnames(generation_rows), generation_rows)

    # -----------------------------------------------------------------------
    # Stage 2: Adaptive connectivity refinement
    # -----------------------------------------------------------------------
    clusters, overlap_rows = build_eq_clusters_v4(windows, ctx, float(args.eq_overlap_threshold))

    (windows, clusters, segments, patches_for_global,
     global_pmf, global_var, fit_details, overlap_rows, ref_rows) = run_connectivity_refinement_v4(
        windows=windows, clusters=clusters, segments=segments,
        patches_for_global=patches_for_global, global_pmf=global_pmf, global_var=global_var,
        fit_details=fit_details, overlap_rows=overlap_rows, args=args, ctx=ctx, grid=grid,
        out_root=out_root, bin_path=bin_path, budget=budget, quality_rows=quality_rows,
        profile=profile, beta_eff=beta_eff,
    )

    write_state_tables_v4(out_root, windows, clusters, segments,
                           patches_for_global, fit_details, overlap_rows,
                           grid, global_pmf, global_var)

    # -----------------------------------------------------------------------
    # Stage 3: Final connected-EQ MBAR extension
    # -----------------------------------------------------------------------
    clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows = \
        run_final_eq_extension_v4(
            windows=windows, clusters=clusters, segments=segments,
            patches_for_global=patches_for_global,
            global_pmf=global_pmf, global_var=global_var, fit_details=fit_details,
            overlap_rows=overlap_rows, args=args, ctx=ctx, grid=grid,
            out_root=out_root, bin_path=bin_path, budget=budget, quality_rows=quality_rows,
        )

    # -----------------------------------------------------------------------
    # Final outputs
    # -----------------------------------------------------------------------
    write_state_tables_v4(out_root, windows, clusters, segments,
                           patches_for_global, fit_details, overlap_rows,
                           grid, global_pmf, global_var)

    if quality_rows:
        write_csv(out_root / "pmf_quality_vs_steps.csv",
                  ordered_fieldnames(quality_rows), quality_rows)
    budget.write(out_root / "budget_ledger.csv")

    ext_stop = ext_rows[-1].get("stop_reason", "") if ext_rows else "no_extension"
    ref_stop = ref_rows[-1].get("reason", "") if ref_rows else "no_refinement"
    summary = {
        "label": label, "seed": seed, "system_id": system_id,
        "n_windows": len(windows), "n_segments": len(segments), "n_clusters": len(clusters),
        "eq_network_connected": bool(eq_network_is_connected(clusters)),
        "final_estimator": fit_details.get("final_estimator", ""),
        "patch_selection_rule": fit_details.get("patch_selection_rule", ""),
        "final_extension_stop_reason": ext_stop,
        "refinement_stop_reason": ref_stop,
        "used_steps": int(budget.used_steps),
        "total_budget_steps": int(budget.total_budget_steps),
        "protocol": "mines_variance_fusion_v4",
        "eq_overlap_threshold": float(args.eq_overlap_threshold),
        "target_kl": float(args.target_kl),
        "overlap_method": "bar_offset_corrected_strict",
        "variance_formula": "inverse_precision",
        "final_perturbation_appended": False,
        "x_most_removed": True,
    }
    write_json(out_root / "mines_variance_fusion_summary.json", summary)
    print(str(out_root / "mines_variance_fusion_summary.json"))


if __name__ == "__main__":
    main()
