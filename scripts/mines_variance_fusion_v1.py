#!/usr/bin/env python3
"""
MiNES v1 — Milestoned Nonequilibrium Switching, Version 1.

Implements the current MiNES protocol:
  - KL-GT child placement with globally fixed endpoint-anchored width profile.
  - Transition segment detection via k0 < 0 and x0 between m_i and m_KL.
  - BAR/MBAR pairwise overlap for EQ connectivity (threshold 0.3, not JSD).
  - Bidirectional NES/MTS without NES truncation.
  - Connected-EQ MBAR as final PMF estimator.

See docs/current_mines_protocol.md for the authoritative protocol reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

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
    mode_x_from_samples,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
    """Globally fixed endpoint-anchored Gaussian width profile.

    Computed once from L0 and R0 endpoints. Not updated per generation.
    Avoids progressive narrowing of child windows.
    """
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


def tail_rows_from_eq_rows(
    eq_rows: list[dict[str, Any]], tail_fraction: float
) -> list[dict[str, Any]]:
    n = len(eq_rows)
    if n == 0:
        return []
    start = max(0, int(math.floor(n * (1.0 - max(0.0, min(1.0, float(tail_fraction)))))))
    return eq_rows[start:]


def eq_tail_samples(window: EnsembleWindow) -> np.ndarray:
    values = [float(r["x"]) for r in window.tail_rows if r.get("x", "") != ""]
    return np.asarray(values, dtype=float)


def cluster_name_from_windows(windows: list[EnsembleWindow]) -> str:
    names = sorted(w.name for w in windows)
    return "cluster__" + "__".join(names)


# ---------------------------------------------------------------------------
# GlobalGTWidthProfile: endpoint-anchored Gaussian width profile
# ---------------------------------------------------------------------------

def make_global_gt_profile(
    left_window: EnsembleWindow, right_window: EnsembleWindow
) -> GlobalGTWidthProfile:
    """Build the globally fixed width profile from first-generation endpoint windows."""
    sigma_L = max(float(left_window.std_x), 1e-6)
    sigma_R = max(float(right_window.std_x), 1e-6)
    return GlobalGTWidthProfile(
        m_L0=float(left_window.mean_x),
        sigma_L0=sigma_L,
        m_R0=float(right_window.mean_x),
        sigma_R0=sigma_R,
    )


# ---------------------------------------------------------------------------
# KL-GT: Gaussian KL distance and target-mean solver
# ---------------------------------------------------------------------------

def gaussian_kl_divergence(
    m_i: float, sigma_i: float, m_j: float, sigma_j: float
) -> float:
    """KL[ N(m_i, sigma_i^2) || N(m_j, sigma_j^2) ].

    = log(sigma_j/sigma_i) + (sigma_i^2 + (m_i - m_j)^2) / (2*sigma_j^2) - 0.5
    """
    if sigma_i <= 0 or sigma_j <= 0:
        return float("inf")
    return (
        math.log(sigma_j / sigma_i)
        + (sigma_i ** 2 + (m_i - m_j) ** 2) / (2.0 * sigma_j ** 2)
        - 0.5
    )


def solve_kl_gt_target(
    m_i: float,
    sigma_i: float,
    profile: GlobalGTWidthProfile,
    target_kl: float,
    direction: str,  # "right" or "left"
    search_bound: float,
    fallback_midpoint: float | None = None,
) -> tuple[float, bool, str]:
    """Solve for m_KL where KL[N(m_i,sigma_i^2) || N(m_KL,sigma_GT(m_KL)^2)] = target_kl.

    Returns (m_KL, fallback_used, fallback_reason).
    """
    sigma_i = max(float(sigma_i), 1e-6)

    def kl_at(m_j: float) -> float:
        sig_j = max(profile.sigma(m_j), 1e-6)
        return gaussian_kl_divergence(m_i, sigma_i, m_j, sig_j)

    if direction == "right":
        lo, hi = float(m_i) + 1e-8, float(search_bound)
    else:
        lo, hi = float(search_bound), float(m_i) - 1e-8
        lo, hi = min(lo, hi), max(lo, hi)

    kl_lo = kl_at(lo)
    kl_hi = kl_at(hi)

    if not (math.isfinite(kl_lo) and math.isfinite(kl_hi)):
        mid = fallback_midpoint if fallback_midpoint is not None else (lo + hi) / 2.0
        return mid, True, "kl_function_not_finite_at_bounds"

    if kl_lo > float(target_kl):
        # Already past target at the closest point — use that point
        m_kl = lo if direction == "right" else hi
        return m_kl, True, "kl_already_exceeds_target_at_lo"

    if kl_hi < float(target_kl):
        # Target KL not reached anywhere in search range
        mid = fallback_midpoint if fallback_midpoint is not None else hi
        return mid, True, "kl_target_not_reached_in_search_range"

    # Binary search: kl_lo <= target_kl <= kl_hi
    for _ in range(64):
        mid = (lo + hi) / 2.0
        kl_mid = kl_at(mid)
        if abs(kl_mid - float(target_kl)) < 1e-8:
            break
        if kl_mid < float(target_kl):
            lo = mid
        else:
            hi = mid

    m_kl = (lo + hi) / 2.0
    return m_kl, False, ""


# ---------------------------------------------------------------------------
# Local harmonic background fit (mean-only, v1)
# ---------------------------------------------------------------------------

def fit_local_harmonic_mean_only(
    left_window: EnsembleWindow, right_window: EnsembleWindow
) -> dict[str, Any]:
    """Estimate mean-only background harmonic from two bracketing EQ windows.

    Solves for k0, x0 from the force-balance condition that each window's mean
    is the equilibrium position under the combined bias + background potential.
    """
    m_L, x_L, k_L = float(left_window.mean_x), float(left_window.center_x), float(left_window.k)
    m_R, x_R, k_R = float(right_window.mean_x), float(right_window.center_x), float(right_window.k)

    denom = m_L - m_R
    if abs(denom) < 1e-12:
        return {"k0": float("nan"), "x0": float("nan"), "fit_valid": False,
                "fit_source": "mean_only", "fit_fallback_reason": "degenerate_m_L_equals_m_R"}

    try:
        k0 = (k_R * (m_R - x_R) - k_L * (m_L - x_L)) / denom
        if abs(k0) < 1e-12:
            return {"k0": k0, "x0": float("nan"), "fit_valid": False,
                    "fit_source": "mean_only", "fit_fallback_reason": "k0_near_zero"}
        x0 = m_L + k_L * (m_L - x_L) / k0
        return {"k0": float(k0), "x0": float(x0), "fit_valid": True,
                "fit_source": "mean_only", "fit_fallback_reason": ""}
    except Exception as e:
        return {"k0": float("nan"), "x0": float("nan"), "fit_valid": False,
                "fit_source": "mean_only", "fit_fallback_reason": str(e)}


# ---------------------------------------------------------------------------
# Child window design: KL-GT + transition detection + bias construction
# ---------------------------------------------------------------------------

def design_child_window(
    *,
    current_window: EnsembleWindow,
    opposite_window: EnsembleWindow,
    profile: GlobalGTWidthProfile,
    k_min: float,
    k_max: float,
    beta_eff: float,
    target_kl: float,
    direction: str,  # "right" (left chain moves right) or "left" (right chain moves left)
) -> dict[str, Any]:
    """Design a child window using KL-GT with transition segment detection.

    Returns a diagnostic dict with all proposal parameters.
    No force matching — uses local harmonic inversion.
    """
    m_i = float(current_window.mean_x)
    sigma_i = max(float(current_window.std_x), 1e-6)
    m_opp = float(opposite_window.mean_x)

    # --- KL-GT target mean ---
    if direction == "right":
        search_bound = m_opp
        fallback_mid = (m_i + m_opp) / 2.0
    else:
        search_bound = m_opp
        fallback_mid = (m_i + m_opp) / 2.0

    m_KL, kl_fallback, kl_fallback_reason = solve_kl_gt_target(
        m_i, sigma_i, profile, target_kl, direction, search_bound, fallback_mid
    )

    # Clamp m_KL toward opposite to prevent overshoot
    if direction == "right":
        m_KL = min(m_KL, m_opp)
        m_KL = max(m_KL, m_i + 1e-6)
    else:
        m_KL = max(m_KL, m_opp)
        m_KL = min(m_KL, m_i - 1e-6)

    # Predicted KL at solved m_KL (for diagnostics)
    sig_KL = max(profile.sigma(m_KL), 1e-6)
    predicted_kl = gaussian_kl_divergence(m_i, sigma_i, m_KL, sig_KL)

    # --- Local harmonic background from bracketing windows ---
    bg = fit_local_harmonic_mean_only(
        current_window if direction == "right" else opposite_window,
        opposite_window if direction == "right" else current_window,
    )
    k0 = float(bg["k0"]) if bg["fit_valid"] and math.isfinite(bg["k0"]) else float("nan")
    x0 = float(bg["x0"]) if bg["fit_valid"] and math.isfinite(bg["x0"]) else float("nan")
    bg_fit_valid = bool(bg["fit_valid"])

    # --- Transition segment detection ---
    # A step is a transition segment (barrier-like) if:
    #   k0 < 0  AND  min(m_i, m_KL) < x0 < max(m_i, m_KL)
    transition_segment = (
        bg_fit_valid
        and math.isfinite(k0) and math.isfinite(x0)
        and k0 < 0.0
        and min(m_i, m_KL) < x0 < max(m_i, m_KL)
    )

    if transition_segment:
        # Reflected target mean: ignore KL target, use barrier reflection
        # Future safeguard: the reflected target m_next = 2*x0 - m_i can jump too far if the
        # local negative-curvature fit is noisy. If this becomes unstable, add a maximum
        # reflected displacement or clip m_next to a trusted local interval. For now, do
        # not cap the reflected target.
        m_next = 2.0 * x0 - m_i
        proposal_rule = "barrier_reflection"
    else:
        # Basin-like step: use KL-GT target
        m_next = m_KL
        proposal_rule = "kl_gt_basin"

    sigma_next = max(profile.sigma(m_next), 1e-6)

    # --- Bias parameter construction ---
    # k_raw = 1 / (beta_eff * sigma_next^2) - k0
    # x_raw = ((k0 + k_raw) * m_next - k0 * x0) / k_raw
    # Priority: (1) preserve m_next, (2) keep k_child in [k_min, k_max], (3) match sigma_next
    k0_for_bias = k0 if (bg_fit_valid and math.isfinite(k0)) else 0.0
    x0_for_bias = x0 if (bg_fit_valid and math.isfinite(x0)) else m_next

    k_raw = 1.0 / (beta_eff * sigma_next ** 2) - k0_for_bias
    if abs(k_raw) < 1e-12:
        k_raw = float(k_min)

    # Compute x_raw to preserve m_next under the raw spring
    if abs(k_raw) > 1e-12:
        x_raw = ((k0_for_bias + k_raw) * m_next - k0_for_bias * x0_for_bias) / k_raw
    else:
        x_raw = m_next

    # Clip k to allowed bounds
    k_clipped = not (float(k_min) <= k_raw <= float(k_max))
    k_child = float(np.clip(k_raw, float(k_min), float(k_max)))

    # Recompute x_child to preserve m_next after k clipping (priority 1)
    if abs(k_child) > 1e-12:
        x_child = ((k0_for_bias + k_child) * m_next - k0_for_bias * x0_for_bias) / k_child
    else:
        x_child = m_next

    sigma_rule_preserved = (not k_clipped)

    return {
        "side": direction,
        "parent_window": current_window.name,
        "opposite_window": opposite_window.name,
        "proposal_rule": proposal_rule,
        "m_i": float(m_i),
        "sigma_i": float(sigma_i),
        "m_KL": float(m_KL),
        "m_next": float(m_next),
        "sigma_next": float(sigma_next),
        "k0": float(k0) if math.isfinite(k0) else "",
        "x0": float(x0) if math.isfinite(x0) else "",
        "transition_segment": int(transition_segment),
        "k_raw": float(k_raw),
        "k_child": float(k_child),
        "k_clipped": int(k_clipped),
        "x_raw": float(x_raw),
        "x_child": float(x_child),
        "sigma_rule_preserved": int(sigma_rule_preserved),
        "fallback_used": int(kl_fallback),
        "fallback_reason": kl_fallback_reason,
        "target_kl": float(target_kl),
        "predicted_kl": float(predicted_kl),
        "bg_fit_valid": int(bg_fit_valid),
        "bg_fit_source": str(bg["fit_source"]),
        "bg_fit_fallback_reason": str(bg["fit_fallback_reason"]),
    }


# ---------------------------------------------------------------------------
# Pairwise BAR/MBAR overlap for EQ connectivity
# ---------------------------------------------------------------------------

def compute_bar_mbar_overlap(
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Compute pairwise BAR/MBAR overlap between two neighboring EQ windows.

    O_ij = mean_{x ~ L} [ 1 / (1 + exp(beta*(u_R(x) - u_L(x)))) ]
    O_ji = mean_{x ~ R} [ 1 / (1 + exp(beta*(u_L(x) - u_R(x)))) ]
    O_pair = min(O_ij, O_ji)   (conservative pairwise overlap)

    where u_i(x) = beta * 0.5 * k_i * (x - cx_i)^2 is the harmonic bias energy.
    """
    _FAIL: dict[str, Any] = {
        "O_ij": float("nan"), "O_ji": float("nan"), "O_pair": float("nan"),
        "bar_solved": False, "bar_delta_f": "", "bar_delta_f_unc": "", "bar_reason": "",
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

        # Reduced bias energies (unitless, divided by kT)
        def u_L(x: np.ndarray) -> np.ndarray:
            return beta * 0.5 * k_L * (x - cx_L) ** 2

        def u_R(x: np.ndarray) -> np.ndarray:
            return beta * 0.5 * k_R * (x - cx_R) ** 2

        # Sigmoid-based BAR/MBAR overlap
        # O_ij = mean_{x~L} sigmoid(u_L(x) - u_R(x))
        # = mean_{x~L} 1/(1 + exp(u_R(x) - u_L(x)))
        delta_L = u_R(x_L) - u_L(x_L)   # perturbation work L → R on L-samples
        delta_R = u_L(x_R) - u_R(x_R)   # perturbation work R → L on R-samples

        O_ij = float(np.mean(1.0 / (1.0 + np.exp(np.clip(delta_L, -500, 500)))))
        O_ji = float(np.mean(1.0 / (1.0 + np.exp(np.clip(delta_R, -500, 500)))))
        O_pair = min(O_ij, O_ji)

        # Also compute BAR delta_f for diagnostics
        w_F = u_R(x_L) - u_L(x_L)
        w_R = u_L(x_R) - u_R(x_R)
        cft = solve_segment_cft_delta_f_once(
            w_F.reshape(-1, 1), w_R.reshape(-1, 1), kT=kT
        )
        bar_solved = bool(cft.get("cft_solved", False))
        return {
            "O_ij": float(O_ij),
            "O_ji": float(O_ji),
            "O_pair": float(O_pair),
            "bar_solved": bar_solved,
            "bar_delta_f": cft.get("delta_f", "") if bar_solved else "",
            "bar_delta_f_unc": cft.get("delta_f_unc", "") if bar_solved else "",
            "bar_reason": str(cft.get("reason", "")),
        }
    except Exception as exc:
        return {**_FAIL, "bar_reason": f"exception: {exc}"}


# ---------------------------------------------------------------------------
# EQ clustering: BAR/MBAR overlap-based (not JSD)
# ---------------------------------------------------------------------------

def build_eq_clusters_v1(
    windows: list[EnsembleWindow],
    ctx: dict[str, Any],
    overlap_threshold: float,
) -> tuple[list[EQCluster], list[dict[str, Any]]]:
    """Merge neighboring EQ windows into clusters by BAR/MBAR pairwise overlap.

    Neighboring windows are connected if O_pair >= overlap_threshold (default 0.3).
    JSD is written as a diagnostic column only and does not drive connectivity.
    """
    ordered = sorted(windows, key=lambda w: (float(w.mean_x), str(w.name)))
    if not ordered:
        return [], []

    clusters: list[EQCluster] = []
    overlap_rows: list[dict[str, Any]] = []
    current: list[EnsembleWindow] = [ordered[0]]

    for window in ordered[1:]:
        left_window = current[-1]
        overlap = compute_bar_mbar_overlap(left_window, window, ctx)

        # JSD as diagnostic only — not used for connectivity
        x_L = eq_tail_samples(left_window)
        x_R = eq_tail_samples(window)
        grid_dx = float(ctx.get("grid_dx", 0.1))
        grid_lo = min(float(np.nanmin(x_L)), float(np.nanmin(x_R))) - 2 * grid_dx
        grid_hi = max(float(np.nanmax(x_L)), float(np.nanmax(x_R))) + 2 * grid_dx
        jsd_grid = np.arange(grid_lo, grid_hi + grid_dx, grid_dx)
        try:
            jsd_raw = float(pair_js_divergence(x_L, x_R, jsd_grid))
            jsd_norm = float(np.sqrt(max(0.0, jsd_raw / math.log(2.0))))
        except Exception:
            jsd_raw = float("nan")
            jsd_norm = float("nan")

        connected = math.isfinite(float(overlap["O_pair"])) and float(overlap["O_pair"]) >= float(overlap_threshold)

        row = {
            "left_window": left_window.name,
            "right_window": window.name,
            "left_mean_x": float(left_window.mean_x),
            "right_mean_x": float(window.mean_x),
            "left_center_x": float(left_window.center_x),
            "right_center_x": float(window.center_x),
            "O_ij": overlap["O_ij"],
            "O_ji": overlap["O_ji"],
            "O_pair": overlap["O_pair"],
            "eq_overlap_threshold": float(overlap_threshold),
            "connected": int(connected),
            "bar_delta_f": overlap["bar_delta_f"],
            "bar_delta_f_unc": overlap["bar_delta_f_unc"],
            "bar_solved": int(overlap["bar_solved"]),
            "bar_reason": overlap["bar_reason"],
            "pair_jsd_diagnostic": float(jsd_norm),
            "cluster_order_coordinate": "mean_x",
        }
        overlap_rows.append(row)

        if connected:
            current.append(window)
        else:
            cluster_windows = sorted(current, key=lambda w: (float(w.mean_x), str(w.name)))
            clusters.append(EQCluster(
                name=cluster_name_from_windows(cluster_windows),
                windows=cluster_windows,
                left_x=float(min(w.mean_x for w in cluster_windows)),
                right_x=float(max(w.mean_x for w in cluster_windows)),
            ))
            current = [window]

    cluster_windows = sorted(current, key=lambda w: (float(w.mean_x), str(w.name)))
    clusters.append(EQCluster(
        name=cluster_name_from_windows(cluster_windows),
        windows=cluster_windows,
        left_x=float(min(w.mean_x for w in cluster_windows)),
        right_x=float(max(w.mean_x for w in cluster_windows)),
    ))
    return clusters, overlap_rows


def eq_network_is_connected(clusters: list[EQCluster]) -> bool:
    return len(clusters) == 1


# ---------------------------------------------------------------------------
# EQ window sampling
# ---------------------------------------------------------------------------

def run_eq_window_v1(
    *,
    name: str,
    center_x: float,
    k: float,
    generation: int,
    side: str,
    bin_path: str,
    ctx: dict[str, Any],
    n_eq_steps: int,
    eq_save_every: int,
    tail_fraction: float,
    seed: int,
    root: Path,
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
    # Use grid from ctx if available for mode estimate
    grid_dx = float(ctx.get("grid_dx", 0.1))
    mode_grid = np.arange(float(np.nanmin(tail_x_finite)), float(np.nanmax(tail_x_finite)) + grid_dx, grid_dx)
    x_most = float(mode_x_from_samples(tail_x_finite, mode_grid)) if mode_grid.size > 0 else mean_x
    window = EnsembleWindow(
        name=name, center_x=float(center_x), k=float(k),
        root=root, eq_file=eq_file, tail_file=tail_file,
        eq_rows=eq_rows, tail_rows=tail_rows,
        mean_x=mean_x, std_x=std_x, x_most=x_most,
        generation=int(generation), side=side,
    )
    write_json(root / "window_summary.json", {
        "name": name, "center_x": center_x, "k": k,
        "mean_x": mean_x, "std_x": std_x, "generation": generation, "side": side,
    })
    return window


# ---------------------------------------------------------------------------
# NEQ bridge protocol: linear interpolation (no NES truncation)
# ---------------------------------------------------------------------------

def build_linear_bridge_protocol_v1(
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    n_time: int,
    k_min: float,
    k_max: float,
) -> dict[str, Any]:
    """Linear interpolation of umbrella center and sqrt(k) along the NEQ path."""
    n_time = max(2, int(n_time))
    cx_L, k_L = float(left_window.center_x), float(left_window.k)
    cx_R, k_R = float(right_window.center_x), float(right_window.k)
    s_values = np.linspace(0.0, 1.0, n_time)
    centers = [float(cx_L * (1 - s) + cx_R * s) for s in s_values]
    sqrt_ks = [math.sqrt(k_L) * (1 - s) + math.sqrt(k_R) * s for s in s_values]
    ks = [float(np.clip(sk ** 2, k_min, k_max)) for sk in sqrt_ks]
    return {"centers": centers, "ks": ks}


def run_neq_segment_v1(
    *,
    name: str,
    left_boundary: EnsembleWindow,
    right_boundary: EnsembleWindow,
    left_source: EQCluster | EnsembleWindow,
    right_source: EQCluster | EnsembleWindow,
    bin_path: str,
    ctx: dict[str, Any],
    t_neq: int,
    n_neq_traj: int,
    seed: int,
    root: Path,
    k_min: float,
    k_max: float,
    neq_pair_source: str = "newly_generated",
) -> NEQSegment:
    """Run bidirectional NES between two boundary windows. No NES truncation."""
    root.mkdir(parents=True, exist_ok=True)
    n_time = max(2, int(t_neq))
    protocol = build_linear_bridge_protocol_v1(
        left_boundary, right_boundary, n_time, k_min, k_max,
    )
    centers_fwd = protocol["centers"]
    ks_fwd = protocol["ks"]
    centers_rev = list(reversed(centers_fwd))
    ks_rev = list(reversed(ks_fwd))

    fwd_path_file = root / "protocol_forward.csv"
    rev_path_file = root / "protocol_reverse.csv"
    _write_protocol_path(fwd_path_file, centers_fwd, ks_fwd)
    _write_protocol_path(rev_path_file, centers_rev, ks_rev)

    fwd_root = root / "forward"
    rev_root = root / "reverse"
    fwd_root.mkdir(parents=True, exist_ok=True)
    rev_root.mkdir(parents=True, exist_ok=True)

    neq_nout = max(1, int(math.ceil(float(t_neq) / 100.0)))
    protocol_k_fwd = float(np.mean([abs(k) for k in ks_fwd]))
    k_midscale = float(ctx.get("nes_screen", {}).get("fixed", {}).get("k_midscale", 1.0))

    for direction, eq_left, eq_right, cx_L, cx_R, fpath, out_dir in [
        ("fwd", left_boundary.eq_file, right_boundary.eq_file,
         float(left_boundary.center_x), float(right_boundary.center_x),
         fwd_path_file, fwd_root),
        ("rev", right_boundary.eq_file, left_boundary.eq_file,
         float(right_boundary.center_x), float(left_boundary.center_x),
         rev_path_file, rev_root),
    ]:
        cmd = [
            bin_path,
            *build_common_args(ctx),
            "-k", str(protocol_k_fwd),
            "-k_midscale", str(k_midscale),
            "-A_center", f"{cx_L},0.0",
            "-B_center", f"{cx_R},0.0",
            "-eq0", str(eq_left),
            "-eq1", str(eq_right),
            "-fpath", str(fpath),
            "-N_neq", str(n_neq_traj),
            "-T_neq", str(t_neq),
            "-neq_nout", str(neq_nout),
            "-neq_seed", str(seed if direction == "fwd" else seed + 1),
            "-out_dir", str(out_dir),
            "-log", str(out_dir / "neq.log"),
        ]
        run_checked(cmd)

    # Read trajectory files
    def read_traj_dir(d: Path) -> tuple[list[list[dict[str, str]]], list[Path]]:
        trajs: list[list[dict[str, str]]] = []
        files: list[Path] = sorted(d.glob("neq_*.csv"))
        for f in files:
            rows = read_csv_rows(f)
            trajs.append(rows)
        return trajs, files

    fwd_trajs, fwd_files = read_traj_dir(fwd_root)
    rev_trajs, rev_files = read_traj_dir(rev_root)

    segment = NEQSegment(
        name=name,
        left=left_source, right=right_source,
        left_boundary=left_boundary, right_boundary=right_boundary,
        root=root,
        forward_trajectories=fwd_trajs,
        reverse_trajectories=rev_trajs,
        forward_trajectory_files=fwd_files,
        reverse_trajectory_files=rev_files,
        forward_path_file=fwd_path_file,
        reverse_path_file=rev_path_file,
        protocol_k=protocol_k_fwd,
        protocol_mode="linear_v1",
        connectivity={"neq_pair_source": neq_pair_source, "final_perturbation_appended": 1},
        mts_patch_built=False,
        cft_summary={},
    )
    return segment


# ---------------------------------------------------------------------------
# EQ cluster patch (EQ-MBAR)
# ---------------------------------------------------------------------------

def build_eq_cluster_patch_v1(
    cluster: EQCluster,
    grid: np.ndarray,
    ctx: dict[str, Any],
    n_boot: int,
    patch_root: Path,
    rng_seed: int,
) -> PMFPatch:
    patch_dir = patch_root / cluster.name
    patch_dir.mkdir(parents=True, exist_ok=True)
    window_rows = [
        {"tail_x": eq_tail_samples(w), "x_m": float(w.center_x), "k_m": float(w.k), "name": w.name}
        for w in cluster.windows
    ]
    base_pmf, ess, probability = direct_eq_mbar_pmf(window_rows, grid, ctx)
    # Shift to min=0 over finite bins
    finite_mask = np.isfinite(base_pmf)
    if np.any(finite_mask):
        base_pmf[finite_mask] -= float(np.nanmin(base_pmf[finite_mask]))

    variance_stack = []
    anchor_variances: dict[str, np.ndarray] = {}
    for ai, window in enumerate(cluster.windows):
        boot = bootstrap_direct_eq_mbar(
            window_rows, grid, float(window.mean_x), int(rng_seed + 31 * ai), ctx, int(n_boot)
        )
        bv = np.asarray(boot["boot_var"], dtype=float)
        variance_stack.append(bv)
        anchor_variances[window.name] = bv

    variance = (
        np.nanmin(np.vstack(variance_stack), axis=0) if variance_stack
        else np.full(len(grid), np.nan, dtype=float)
    )
    all_tail = np.concatenate([eq_tail_samples(w) for w in cluster.windows])
    coverage = np.isfinite(base_pmf) & np.isfinite(variance) & coverage_mask_from_samples(all_tail, grid)

    anchor_variances["var_eq_min"] = variance
    return PMFPatch(
        name=cluster.name, kind="EQ_MBAR", root=patch_dir,
        grid=np.asarray(grid, dtype=float), pmf=base_pmf,
        variance=variance, coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[w.name for w in cluster.windows],
        metadata={
            "cluster_name": cluster.name,
            "window_names": [w.name for w in cluster.windows],
            "left_x": float(cluster.left_x), "right_x": float(cluster.right_x),
            "n_boot": int(n_boot),
        },
        anchor_variances=anchor_variances,
    )


# ---------------------------------------------------------------------------
# NEQ/MTS patch (bidirectional NES + MTS, no NES truncation)
# ---------------------------------------------------------------------------

def read_protocol_centers_and_k(path_file: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path_file)
    centers = np.asarray([float(r["center_x"]) for r in rows], dtype=float)
    ks = np.asarray([float(r["k"]) for r in rows], dtype=float)
    return centers, ks


def build_neq_mts_patch_v1(
    segment: NEQSegment,
    grid: np.ndarray,
    ctx: dict[str, Any],
    n_boot: int,
    patch_root: Path,
    rng_seed: int,
) -> PMFPatch:
    """Build NEQ/MTS patch. No NES truncation — uses full trajectories."""
    patch_dir = patch_root / segment.name
    patch_dir.mkdir(parents=True, exist_ok=True)

    fwd_frames = [pd.DataFrame(rows) for rows in segment.forward_trajectories]
    rev_frames = [pd.DataFrame(rows) for rows in segment.reverse_trajectories]
    x_fwd, work_fwd = trajectory_frames_to_arrays(fwd_frames)
    x_rev, work_rev = trajectory_frames_to_arrays(rev_frames)
    centers, ks = read_protocol_centers_and_k(segment.forward_path_file)

    n_time = min(
        x_fwd.shape[1] if x_fwd.ndim == 2 else 0,
        work_fwd.shape[1] if work_fwd.ndim == 2 else 0,
        x_rev.shape[1] if x_rev.ndim == 2 else 0,
        work_rev.shape[1] if work_rev.ndim == 2 else 0,
        len(centers), len(ks),
    )
    centers = centers[:n_time]
    ks = ks[:n_time]

    kT = float(ctx.get("thermal_kT", 1.0))
    cft = solve_segment_cft_delta_f_once(
        work_fwd[:, :n_time], work_rev[:, :n_time], kT=kT
    )
    segment.cft_summary = {
        "cft_solved": bool(cft.get("cft_solved", False)),
        "delta_f": cft.get("delta_f"),
        "delta_f_unc": cft.get("delta_f_unc"),
        "method": cft.get("method"),
        "reason": cft.get("reason"),
        "mts_solved": 1,
        "neq_pair_source": segment.connectivity.get("neq_pair_source", ""),
        "final_perturbation_appended": segment.connectivity.get("final_perturbation_appended", 1),
    }

    fixed_delta_f = (
        float(cft["delta_f"])
        if bool(cft.get("cft_solved", False)) and cft.get("delta_f") is not None
        and math.isfinite(float(cft["delta_f"]))
        else None
    )
    left_ref_x = float(segment.left_boundary.mean_x)

    boot_result = bootstrap_bidirectional_mts_pmf(
        x_fwd, work_fwd, x_rev, work_rev, centers, ks, grid,
        reference_x=left_ref_x, kT=kT,
        n_boot=int(n_boot), fk_boot=max(int(n_boot // 8), 4),
        rng_seed=int(rng_seed),
        fixed_delta_f=fixed_delta_f,
        recompute_delta_f_per_bootstrap=(fixed_delta_f is None),
    )

    pmf_arr = np.asarray(boot_result["pmf"], dtype=float)
    var_arr = np.asarray(boot_result["boot_var"], dtype=float)
    coverage = coverage_mask_from_samples(
        np.concatenate([x_fwd.ravel(), x_rev.ravel()]), grid
    ) & np.isfinite(pmf_arr) & np.isfinite(var_arr)

    segment.mts_patch_built = True
    return PMFPatch(
        name=segment.name, kind="NEQ_MTS", root=patch_dir,
        grid=np.asarray(grid, dtype=float), pmf=pmf_arr,
        variance=var_arr, coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[segment.left_boundary.name, segment.right_boundary.name],
        metadata={
            "segment_name": segment.name, "cft_solved": bool(cft.get("cft_solved", False)),
            "delta_f": cft.get("delta_f"), "n_boot": int(n_boot),
            "neq_pair_source": segment.connectivity.get("neq_pair_source", ""),
            "mts_solved": 1,
        },
    )


# ---------------------------------------------------------------------------
# PMF fusion: inverse-variance-weighted least-squares
# ---------------------------------------------------------------------------

def fit_global_pmf_v1(
    patches: list[PMFPatch], grid: np.ndarray, variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """min_{G,c} Σ_p Σ_{x in S_p} (G(x) - F_p(x) - c_p)^2 / (var_p(x) + eps)."""
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
        return nan.copy(), nan.copy(), {"n_patches": len(patches), "n_observations": 0,
                                        "patch_offsets": {p.name: None for p in patches}}

    n_grid, n_patch = len(grid), len(patches)
    gauge_idx = int(obs[0][1])
    n_rows = len(obs) + 1
    A = np.zeros((n_rows, n_grid + n_patch), dtype=float)
    b = np.zeros(n_rows, dtype=float)
    for ri, (pi, gi, fval, w) in enumerate(obs):
        s = math.sqrt(w)
        A[ri, gi] = s
        A[ri, n_grid + pi] = -s
        b[ri] = s * fval
    A[-1, gauge_idx] = 1e6
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    global_pmf = np.asarray(sol[:n_grid], dtype=float)
    offsets = np.asarray(sol[n_grid:], dtype=float)
    # Shift min to 0
    finite = np.isfinite(global_pmf)
    if np.any(finite):
        shift = float(np.nanmin(global_pmf[finite]))
        global_pmf[finite] -= shift
        offsets -= shift

    # Global variance: mean patch variance at each bin, weighted by coverage count
    global_var = np.full(n_grid, np.nan, dtype=float)
    n_cover = np.zeros(n_grid, dtype=int)
    var_sum = np.zeros(n_grid, dtype=float)
    for pi, gi, _, w in obs:
        n_cover[gi] += 1
        var_sum[gi] += float(patches[pi].variance[gi])
    covered = n_cover > 0
    global_var[covered] = var_sum[covered] / n_cover[covered]

    rms = float(np.sqrt(np.mean([(b[ri] / math.sqrt(w) - (global_pmf[gi] - offsets[pi])) ** 2
                                  for ri, (pi, gi, _, w) in enumerate(obs)]))) if obs else float("nan")
    return global_pmf, global_var, {
        "n_patches": len(patches), "n_observations": len(obs),
        "patch_offsets": {patches[pi].name: float(offsets[pi]) for pi in range(n_patch)},
        "rms_residual": rms,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_neighbor_eq_overlap(out_root: Path, overlap_rows: list[dict[str, Any]]) -> None:
    _COLS = [
        "left_window", "right_window", "left_mean_x", "right_mean_x",
        "left_center_x", "right_center_x", "O_ij", "O_ji", "O_pair",
        "eq_overlap_threshold", "connected", "bar_delta_f", "bar_delta_f_unc",
        "bar_solved", "bar_reason", "pair_jsd_diagnostic", "cluster_order_coordinate",
    ]
    write_csv(out_root / "neighbor_eq_overlap.csv", _COLS, overlap_rows)


def write_state_tables_v1(
    base_root: Path,
    windows: list[EnsembleWindow],
    clusters: list[EQCluster],
    segments: list[NEQSegment],
    patches: list[PMFPatch],
    fit_details: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
    grid: np.ndarray,
    global_pmf: np.ndarray,
    global_var: np.ndarray,
) -> None:
    base_root.mkdir(parents=True, exist_ok=True)

    write_csv(base_root / "windows.csv", [
        "name", "center_x", "k", "mean_x", "std_x", "x_most", "generation", "side",
    ], [
        {"name": w.name, "center_x": w.center_x, "k": w.k, "mean_x": w.mean_x,
         "std_x": w.std_x, "x_most": w.x_most, "generation": w.generation, "side": w.side}
        for w in windows
    ])

    write_csv(base_root / "clusters.csv", [
        "name", "n_windows", "left_x", "right_x", "window_names",
    ], [
        {"name": c.name, "n_windows": len(c.windows),
         "left_x": c.left_x, "right_x": c.right_x,
         "window_names": ";".join(w.name for w in c.windows)}
        for c in clusters
    ])

    write_csv(base_root / "segments.csv", [
        "name", "left_boundary", "right_boundary", "protocol_mode", "protocol_k",
        "mts_patch_built", "cft_solved",
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

    # global_pmf.csv
    pmf_rows = [
        {"x": float(grid[i]), "global_pmf": float(global_pmf[i]) if np.isfinite(global_pmf[i]) else "",
         "global_variance": float(global_var[i]) if np.isfinite(global_var[i]) else ""}
        for i in range(len(grid))
    ]
    write_csv(base_root / "global_pmf.csv", ["x", "global_pmf", "global_variance"], pmf_rows)

    fd = dict(fit_details)
    fd["eq_network_connected"] = bool(eq_network_is_connected(clusters))
    fd["final_estimator"] = "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
    write_json(base_root / "global_fit_summary.json", fd)


# ---------------------------------------------------------------------------
# Smoke-test overrides
# ---------------------------------------------------------------------------

def apply_quick_test_overrides(args: argparse.Namespace) -> None:
    args.n_eq_steps = 1000
    args.t_neq = 200
    args.n_neq_traj = 10
    args.max_generations = 1
    args.n_bootstrap_eq = 8
    args.n_bootstrap_neq = 8
    args.final_refinement_mode = "none"
    args.total_budget_steps = 20000


# ---------------------------------------------------------------------------
# PMF quality tracking (simplified)
# ---------------------------------------------------------------------------

def compute_pmf_quality_v1(
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
# Final EQ-extension (connected-EQ MBAR only)
# ---------------------------------------------------------------------------

def run_final_eq_extension_v1(
    *,
    windows: list[EnsembleWindow],
    clusters: list[EQCluster],
    segments: list[NEQSegment],
    patches_for_global: list[PMFPatch],
    global_pmf: np.ndarray,
    global_var: np.ndarray,
    fit_details: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    ctx: dict[str, Any],
    grid: np.ndarray,
    out_root: Path,
    bin_path: str,
    budget: BudgetTracker,
    quality_rows: list[dict[str, Any]],
) -> tuple[list[EQCluster], list[PMFPatch], np.ndarray, np.ndarray, dict[str, Any],
           list[dict[str, Any]], list[dict[str, Any]]]:
    _COLS = [
        "round", "used_steps", "remaining_steps", "n_selected_windows",
        "eq_extension_steps", "round_cost", "max_mbar_ddf", "x_at_max_mbar_ddf",
        "target_mbar_ddf", "stop_reason", "n_clusters_after_extension",
    ]
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
        write_csv(out_root / "final_eq_extension_summary.csv", _COLS, ext_rows)

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
            max_ddf = float("nan")
            x_at_max = float("nan")

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

        # Extend each window
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
            ext_rows_csv = read_csv_rows(ext_win_root / "eq_window.csv")
            ext_tail = tail_rows_from_eq_rows(ext_rows_csv, float(args.tail_fraction))
            combined_tail = list(window.tail_rows) + ext_tail
            window.tail_rows = combined_tail
            tail_x = np.asarray([float(r["x"]) for r in combined_tail if r.get("x", "") != ""], dtype=float)
            tail_x = tail_x[np.isfinite(tail_x)]
            if tail_x.size >= 2:
                window.mean_x = float(np.mean(tail_x))
                window.std_x = float(np.std(tail_x, ddof=1))

        budget.spend(round_cost, f"final_eq_extension_round_{round_index:03d}", "EQ_EXTENSION", "final_eq_extension")

        # Rebuild clusters and patches
        new_clusters, new_overlap = build_eq_clusters_v1(windows, ctx, float(args.eq_overlap_threshold))
        overlap_rows = new_overlap

        if not eq_network_is_connected(new_clusters):
            summary_row["stop_reason"] = "eq_connectivity_lost"
            summary_row["n_clusters_after_extension"] = len(new_clusters)
            ext_rows.append(summary_row)
            write_csv(out_root / "final_eq_extension_summary.csv", _COLS, ext_rows)
            return clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows

        clusters = new_clusters
        patch_root = round_root / "patches"
        new_patches: list[PMFPatch] = []
        for ci, cl in enumerate(clusters):
            new_patches.append(build_eq_cluster_patch_v1(
                cl, grid, ctx, int(args.n_bootstrap_eq), patch_root,
                rng_seed=base_seed + 300000 + round_index * 1000 + ci,
            ))
        global_pmf, global_var, fit_details = fit_global_pmf_v1(
            new_patches, grid, float(args.variance_floor)
        )
        patches_for_global = new_patches
        fit_details["patch_selection_rule"] = "connected_EQ_MBAR_only"
        fit_details["eq_network_connected"] = True
        fit_details["final_estimator"] = "connected_EQ_MBAR_only"

        write_state_tables_v1(round_root / "state", windows, clusters, segments,
                               patches_for_global, fit_details, overlap_rows,
                               grid, global_pmf, global_var)
        quality_rows.append(compute_pmf_quality_v1(
            grid, global_pmf, global_var, ctx, int(budget.used_steps),
            f"final_eq_extension_round_{round_index:03d}",
        ))
        summary_row["stop_reason"] = "round_complete"
        ext_rows.append(summary_row)
        write_csv(out_root / "final_eq_extension_summary.csv", _COLS, ext_rows)
        round_index += 1

    write_csv(out_root / "final_eq_extension_summary.csv", _COLS, ext_rows)
    return clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiNES v1 — KL-GT child placement, BAR/MBAR EQ connectivity, no NES truncation."
    )
    parser.add_argument("--system-root", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--label", default="mines_variance_fusion_v1")
    parser.add_argument("--total-budget-steps", default=2500000, type=int)
    parser.add_argument("--t-neq", default=5000, type=int)
    parser.add_argument("--n-neq-traj", default=100, type=int)
    parser.add_argument("--n-eq-steps", default=10000, type=int)
    parser.add_argument("--eq-save-every", default=10, type=int)
    parser.add_argument("--tail-fraction", default=0.9, type=float)
    parser.add_argument("--target-kl", default=1.0, type=float,
                        help="Target directional Gaussian KL distance for basin-like KL-GT exploration steps.")
    parser.add_argument("--eq-overlap-threshold", default=0.3, type=float,
                        help="Pairwise BAR/MBAR overlap cutoff for merging neighboring EQ windows.")
    parser.add_argument("--k-min", default=1.0, type=float)
    parser.add_argument("--k-max", default=100.0, type=float)
    parser.add_argument("--k-rescue", default=10.0, type=float)
    parser.add_argument("--bin-width", default=0.1, type=float)
    parser.add_argument("--max-generations", default=10, type=int)
    parser.add_argument("--n-bootstrap-eq", default=64, type=int)
    parser.add_argument("--n-bootstrap-neq", default=64, type=int)
    parser.add_argument("--variance-floor", default=1e-6, type=float)
    parser.add_argument("--pmf-method", choices=["neq", "eq", "hybrid"], default="neq",
                        help="Provisional PMF estimator (State A only; overridden to eq in connected State B).")
    parser.add_argument("--final-refinement-mode", choices=["none", "eq-extend"], default="eq-extend")
    parser.add_argument("--target-mbar-ddf", default=1e-3, type=float)
    parser.add_argument("--eq-extension-steps", default=None, type=int)
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
    label = str(args.label)
    seed = int(args.seed)

    out_root = system_root / "MINES" / label / "raw" / f"seed_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    grid_dx = float(args.bin_width)
    grid = build_grid(ctx, grid_dx)
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

    # --- Sample L0 and R0 endpoints ---
    eq0_cost = eq_cost(args.n_eq_steps)
    for _ in range(2):
        if not budget.can_spend(eq0_cost):
            raise RuntimeError("Budget too small for endpoint EQ sampling.")
        budget.spend(eq0_cost, "endpoint", "EQ", "init")

    left0 = run_eq_window_v1(
        name="L_gen0", center_x=float(ctx["mines_screen"]["fixed"]["start_x_left"]),
        k=float(args.k_min), generation=0, side="left",
        bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
        eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
        seed=seed + 1, root=out_root / "windows" / "L_gen0",
    )
    right0 = run_eq_window_v1(
        name="R_gen0", center_x=float(ctx["mines_screen"]["fixed"]["start_x_right"]),
        k=float(args.k_min), generation=0, side="right",
        bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
        eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
        seed=seed + 2, root=out_root / "windows" / "R_gen0",
    )
    windows.extend([left0, right0])

    # Build globally fixed GT width profile from L0 and R0
    profile = make_global_gt_profile(left0, right0)

    left_frontier = left0
    right_frontier = right0

    # --- Exploration loop ---
    for generation in range(int(args.max_generations)):
        # Rebuild clusters after each round
        clusters, overlap_rows = build_eq_clusters_v1(windows, ctx, float(args.eq_overlap_threshold))
        if eq_network_is_connected(clusters):
            print(f"[v1] EQ network connected at generation {generation}. Stopping exploration.")
            break

        # Run NEQ between frontier pair
        seg_name = f"seg_L{left_frontier.name}_R{right_frontier.name}_gen{generation}"
        neq_c = neq_cost(args.n_neq_traj, args.t_neq)
        if not budget.can_spend(neq_c):
            print("[v1] Budget exhausted before NEQ at generation", generation)
            break
        budget.spend(neq_c, seg_name, "NEQ", f"generation_{generation}")
        seg = run_neq_segment_v1(
            name=seg_name, left_boundary=left_frontier, right_boundary=right_frontier,
            left_source=left_frontier, right_source=right_frontier,
            bin_path=bin_path, ctx=ctx, t_neq=args.t_neq, n_neq_traj=args.n_neq_traj,
            seed=seed + 100 + generation, root=out_root / "segments" / seg_name,
            k_min=float(args.k_min), k_max=float(args.k_max),
        )
        segments.append(seg)

        # Build NEQ/MTS patch for diagnostics
        patch_root = out_root / "patches" / f"gen{generation}"
        mts_patch = build_neq_mts_patch_v1(
            seg, grid, ctx, int(args.n_bootstrap_neq), patch_root,
            rng_seed=seed + 200 + generation,
        )

        # Design left child (moves right)
        left_design = design_child_window(
            current_window=left_frontier, opposite_window=right_frontier,
            profile=profile, k_min=float(args.k_min), k_max=float(args.k_max),
            beta_eff=beta_eff, target_kl=float(args.target_kl), direction="right",
        )
        # Design right child (moves left)
        right_design = design_child_window(
            current_window=right_frontier, opposite_window=left_frontier,
            profile=profile, k_min=float(args.k_min), k_max=float(args.k_max),
            beta_eff=beta_eff, target_kl=float(args.target_kl), direction="left",
        )

        gen_eq_cost = 2 * eq_cost(args.n_eq_steps)
        if not budget.can_spend(gen_eq_cost):
            print("[v1] Budget exhausted before EQ children at generation", generation)
            break
        budget.spend(gen_eq_cost, f"children_gen{generation}", "EQ", f"generation_{generation}")

        left_name = f"L_gen{generation + 1}"
        right_name = f"R_gen{generation + 1}"
        left_child = run_eq_window_v1(
            name=left_name, center_x=float(left_design["x_child"]),
            k=float(left_design["k_child"]), generation=generation + 1, side="left",
            bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
            eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
            seed=seed + 300 + generation * 2,
            root=out_root / "windows" / left_name,
        )
        right_child = run_eq_window_v1(
            name=right_name, center_x=float(right_design["x_child"]),
            k=float(right_design["k_child"]), generation=generation + 1, side="right",
            bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
            eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
            seed=seed + 300 + generation * 2 + 1,
            root=out_root / "windows" / right_name,
        )
        windows.extend([left_child, right_child])
        left_frontier = left_child
        right_frontier = right_child

        left_design["generation"] = generation
        right_design["generation"] = generation
        generation_rows.extend([left_design, right_design])

        # Rebuild clusters and build provisional PMF
        clusters, overlap_rows = build_eq_clusters_v1(windows, ctx, float(args.eq_overlap_threshold))
        eq_patches: list[PMFPatch] = []
        for ci, cl in enumerate(clusters):
            eq_patches.append(build_eq_cluster_patch_v1(
                cl, grid, ctx, int(args.n_bootstrap_eq),
                out_root / "patches" / f"gen{generation}" / "eq",
                rng_seed=seed + 400 + generation * 100 + ci,
            ))

        # Patch selection (provisional): connected → EQ only; else follow pmf_method
        if eq_network_is_connected(clusters):
            patches_for_global = eq_patches
            patch_selection_rule = "connected_EQ_MBAR_only"
        else:
            neq_patches = [mts_patch]
            if args.pmf_method == "neq":
                patches_for_global = neq_patches
                patch_selection_rule = "only_NEQ_MTS"
            elif args.pmf_method == "hybrid":
                patches_for_global = eq_patches + neq_patches
                patch_selection_rule = "EQ_MBAR_plus_NEQ_MTS"
            else:  # eq
                patches_for_global = eq_patches
                patch_selection_rule = "EQ_MBAR_only_provisional"

        global_pmf, global_var, fit_details = fit_global_pmf_v1(
            patches_for_global, grid, float(args.variance_floor)
        )
        fit_details["patch_selection_rule"] = patch_selection_rule
        fit_details["eq_network_connected"] = bool(eq_network_is_connected(clusters))
        fit_details["final_estimator"] = (
            "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
        )

        gen_root = out_root / f"generation_{generation:03d}"
        write_state_tables_v1(gen_root, windows, clusters, segments,
                               patches_for_global, fit_details, overlap_rows,
                               grid, global_pmf, global_var)
        quality_rows.append(compute_pmf_quality_v1(
            grid, global_pmf, global_var, ctx, int(budget.used_steps), f"generation_{generation}"
        ))

        if eq_network_is_connected(clusters):
            break

    # Final state
    clusters, overlap_rows = build_eq_clusters_v1(windows, ctx, float(args.eq_overlap_threshold))
    eq_patches = []
    for ci, cl in enumerate(clusters):
        eq_patches.append(build_eq_cluster_patch_v1(
            cl, grid, ctx, int(args.n_bootstrap_eq),
            out_root / "patches" / "final" / "eq",
            rng_seed=seed + 700000 + ci,
        ))
    if eq_network_is_connected(clusters):
        patches_for_global = eq_patches
        patch_selection_rule = "connected_EQ_MBAR_only"
    else:
        neq_patches_all = [
            build_neq_mts_patch_v1(s, grid, ctx, int(args.n_bootstrap_neq),
                                    out_root / "patches" / "final" / "neq",
                                    rng_seed=seed + 800000 + si)
            for si, s in enumerate(segments) if s.forward_trajectories
        ]
        if args.pmf_method == "neq" and neq_patches_all:
            patches_for_global = neq_patches_all
            patch_selection_rule = "only_NEQ_MTS"
        elif args.pmf_method == "hybrid" and neq_patches_all:
            patches_for_global = eq_patches + neq_patches_all
            patch_selection_rule = "EQ_MBAR_plus_NEQ_MTS"
        else:
            patches_for_global = eq_patches
            patch_selection_rule = "EQ_MBAR_only_provisional"

    global_pmf, global_var, fit_details = fit_global_pmf_v1(
        patches_for_global, grid, float(args.variance_floor)
    )
    fit_details["patch_selection_rule"] = patch_selection_rule
    fit_details["eq_network_connected"] = bool(eq_network_is_connected(clusters))
    fit_details["final_estimator"] = (
        "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
    )

    write_state_tables_v1(out_root, windows, clusters, segments,
                           patches_for_global, fit_details, overlap_rows,
                           grid, global_pmf, global_var)

    # Write child proposal diagnostics
    if generation_rows:
        write_csv(out_root / "generation_summary.csv",
                  ordered_fieldnames(generation_rows), generation_rows)

    # Final EQ extension (connected-EQ MBAR only)
    clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows = \
        run_final_eq_extension_v1(
            windows=windows, clusters=clusters, segments=segments,
            patches_for_global=patches_for_global,
            global_pmf=global_pmf, global_var=global_var, fit_details=fit_details,
            overlap_rows=overlap_rows, args=args, ctx=ctx, grid=grid,
            out_root=out_root, bin_path=bin_path, budget=budget, quality_rows=quality_rows,
        )

    # Write final outputs
    write_state_tables_v1(out_root, windows, clusters, segments,
                           patches_for_global, fit_details, overlap_rows,
                           grid, global_pmf, global_var)
    if quality_rows:
        write_csv(out_root / "pmf_quality_vs_steps.csv",
                  ordered_fieldnames(quality_rows), quality_rows)
    budget.write(out_root / "budget_ledger.csv")

    n_connected = int(eq_network_is_connected(clusters))
    ext_stop = ext_rows[-1].get("stop_reason", "") if ext_rows else "no_extension"
    summary = {
        "label": label, "seed": seed,
        "n_windows": len(windows), "n_segments": len(segments), "n_clusters": len(clusters),
        "eq_network_connected": bool(n_connected),
        "final_estimator": fit_details.get("final_estimator", ""),
        "patch_selection_rule": fit_details.get("patch_selection_rule", ""),
        "final_extension_stop_reason": ext_stop,
        "used_steps": int(budget.used_steps),
        "total_budget_steps": int(budget.total_budget_steps),
        "protocol": "mines_variance_fusion_v1",
        "eq_overlap_threshold": float(args.eq_overlap_threshold),
        "target_kl": float(args.target_kl),
    }
    write_json(out_root / "mines_variance_fusion_summary.json", summary)
    print(str(out_root / "mines_variance_fusion_summary.json"))


if __name__ == "__main__":
    main()
