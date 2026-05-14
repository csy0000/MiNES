#!/usr/bin/env python3
"""
MiNES v3 — Milestoned Nonequilibrium Switching, Version 3.

Changes from v2:

  v3-1: Final perturbation work is included in CFT/MTS arrays (augmented protocol).
  v3-2: --max-refinement-rounds; --max-rescue-rounds removed.
  v3-3: Refinement: NES only for transition/barrier-like segments; basin → midpoint EQ.
  v3-4: BAR failure means disconnected; raw sigmoid cannot merge EQ windows.
  v3-5: Rebuild EQ clusters after each child-seeding event during exploration.
  v3-6: CFT/MTS failure in transition refinement falls back to midpoint mean-only GT.
  v3-7: x_most removed from EnsembleWindow and all outputs.
  v3-8: mts_solved checks CFT, finite delta_f, and finite PMF/variance coverage.
  v3-9: Refinement loop processes one disconnected pair per round; restarts on fresh clusters.

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
    """v3-7: x_most removed."""
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
    # v3-1: augmented protocol files include endpoint step
    forward_path_file_augmented: Path | None = None
    reverse_path_file_augmented: Path | None = None
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
    names = sorted(w.name for w in windows)
    return "cluster__" + "__".join(names)


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
    sigma_i = max(float(sigma_i), 1e-6)

    def kl_at(m_j: float) -> float:
        return gaussian_kl_divergence(m_i, sigma_i, m_j, max(profile.sigma(m_j), 1e-6))

    if direction == "right":
        lo, hi = float(m_i) + 1e-8, float(search_bound)
    else:
        lo, hi = min(float(search_bound), float(m_i) - 1e-8), max(float(search_bound), float(m_i) - 1e-8)

    kl_lo, kl_hi = kl_at(lo), kl_at(hi)
    if not (math.isfinite(kl_lo) and math.isfinite(kl_hi)):
        mid = fallback_midpoint if fallback_midpoint is not None else (lo + hi) / 2.0
        return mid, True, "kl_function_not_finite_at_bounds"
    if kl_lo > float(target_kl):
        return (lo if direction == "right" else hi), True, "kl_already_exceeds_target_at_lo"
    if kl_hi < float(target_kl):
        mid = fallback_midpoint if fallback_midpoint is not None else hi
        return mid, True, "kl_target_not_reached_in_search_range"

    for _ in range(64):
        mid = (lo + hi) / 2.0
        kl_mid = kl_at(mid)
        if abs(kl_mid - float(target_kl)) < 1e-8:
            break
        if kl_mid < float(target_kl):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, False, ""


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
            return {"k0": k0, "x0": float("nan"), "fit_valid": False,
                    "fit_source": "mean_only", "fit_fallback_reason": "k0_near_zero"}
        x0 = m_L + k_L * (m_L - x_L) / k0
        return {"k0": float(k0), "x0": float(x0), "fit_valid": True,
                "fit_source": "mean_only", "fit_fallback_reason": ""}
    except Exception as e:
        return {"k0": float("nan"), "x0": float("nan"), "fit_valid": False,
                "fit_source": "mean_only", "fit_fallback_reason": str(e)}


def classify_segment(
    left_boundary: EnsembleWindow, right_boundary: EnsembleWindow
) -> tuple[bool, dict[str, Any]]:
    """Return (is_transition_segment, bg_fit_dict) using mean-only local harmonic fit."""
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

    # v3-5: transition detection uses m_i and m_KL (exploration, criterion 4)
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
# v3-4: BAR-offset-corrected overlap — BAR failure = disconnected
# ---------------------------------------------------------------------------

def compute_bar_mbar_overlap_v3(
    left_window: EnsembleWindow, right_window: EnsembleWindow, ctx: dict[str, Any],
) -> dict[str, Any]:
    """Pairwise BAR-offset-corrected overlap.

    If BAR fails, O_pair = nan, connected = False.
    Raw sigmoid is written as diagnostic only and does NOT decide connectivity.

        O_ij = mean_{x~L} 1/(1 + exp( u_R(x) - u_L(x) - delta_f ))
        O_ji = mean_{x~R} 1/(1 + exp( u_L(x) - u_R(x) + delta_f ))
        O_pair = min(O_ij, O_ji)
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
            # v3-4: BAR failure → disconnected; raw sigmoid not used for connectivity
            return {
                "O_ij": float("nan"), "O_ji": float("nan"), "O_pair": float("nan"),
                "bar_solved": False,
                "bar_delta_f": "", "bar_delta_f_unc": "",
                "overlap_method": "bar_failed_no_overlap_assigned",
                "overlap_reason": str(cft.get("reason", "bar_failed")),
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
            "O_ij": float(O_ij), "O_ji": float(O_ji), "O_pair": float(O_pair),
            "bar_solved": True,
            "bar_delta_f": float(cft["delta_f"]),
            "bar_delta_f_unc": float(cft.get("delta_f_unc", float("nan"))),
            "overlap_method": "bar_offset_corrected",
            "overlap_reason": "ok",
            "raw_sigmoid_O_ij_diagnostic": float(raw_O_ij),
            "raw_sigmoid_O_ji_diagnostic": float(raw_O_ji),
        }
    except Exception as exc:
        return {**_FAIL, "overlap_reason": f"exception: {exc}"}


# ---------------------------------------------------------------------------
# v3-4: EQ clustering — BAR failure means disconnected
# ---------------------------------------------------------------------------

def build_eq_clusters_v3(
    windows: list[EnsembleWindow], ctx: dict[str, Any], overlap_threshold: float,
) -> tuple[list[EQCluster], list[dict[str, Any]]]:
    """Merge neighboring EQ windows by BAR-offset-corrected overlap.

    v3-4: connectivity requires bar_solved AND overlap_method == 'bar_offset_corrected'.
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
        overlap = compute_bar_mbar_overlap_v3(left_window, window, ctx)

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

        # v3-4: strict rule — raw sigmoid cannot merge windows
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
    return clusters, overlap_rows


def eq_network_is_connected(clusters: list[EQCluster]) -> bool:
    return len(clusters) == 1


# ---------------------------------------------------------------------------
# EQ window sampling (v3-7: no x_most)
# ---------------------------------------------------------------------------

def run_eq_window_v3(
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
# NEQ bridge: linear interpolation
# ---------------------------------------------------------------------------

def build_linear_bridge_protocol(
    left_window: EnsembleWindow, right_window: EnsembleWindow,
    n_time: int, k_min: float, k_max: float,
) -> dict[str, Any]:
    n_time = max(2, int(n_time))
    cx_L, k_L = float(left_window.center_x), float(left_window.k)
    cx_R, k_R = float(right_window.center_x), float(right_window.k)
    s_values = np.linspace(0.0, 1.0, n_time)
    centers = [float(cx_L * (1 - s) + cx_R * s) for s in s_values]
    ks = [float(np.clip(
        (math.sqrt(k_L) * (1 - s) + math.sqrt(k_R) * s) ** 2, k_min, k_max
    )) for s in s_values]
    return {"centers": centers, "ks": ks}


# ---------------------------------------------------------------------------
# Final perturbation appending
# ---------------------------------------------------------------------------

def append_final_perturbation(
    trajectories: list[list[dict[str, str]]],
    *, endpoint_center: float, endpoint_k: float,
    last_protocol_center: float, last_protocol_k: float,
    direction_label: str,
) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    """Append dW = U_endpoint(x_T) - U_last(x_T) to each trajectory's final row."""
    augmented: list[list[dict[str, str]]] = []
    summary_rows: list[dict[str, Any]] = []
    for traj_idx, rows in enumerate(trajectories):
        base_row: dict[str, Any] = {
            "direction": direction_label, "traj_index": traj_idx,
            "x_final": "", "old_work": "", "dW_final": "", "new_work": "",
            "endpoint_center": endpoint_center, "endpoint_k": endpoint_k,
            "last_protocol_center": last_protocol_center, "last_protocol_k": last_protocol_k,
            "appended": 0,
        }
        if not rows:
            augmented.append(rows)
            summary_rows.append(base_row)
            continue
        last_row = rows[-1]
        try:
            x_T = float(last_row.get("x", float("nan")))
            W_old = float(last_row.get("work", float("nan")))
        except (ValueError, TypeError):
            augmented.append(rows)
            summary_rows.append(base_row)
            continue
        dW = (0.5 * float(endpoint_k) * (x_T - float(endpoint_center)) ** 2
              - 0.5 * float(last_protocol_k) * (x_T - float(last_protocol_center)) ** 2)
        W_new = W_old + dW
        try:
            old_step = int(last_row.get("step", len(rows)))
        except (ValueError, TypeError):
            old_step = len(rows)
        final_row: dict[str, str] = dict(last_row)
        final_row["step"] = str(old_step + 1)
        final_row["x"] = str(x_T)
        final_row["work"] = str(W_new)
        final_row["final_perturbation"] = "1"
        final_row["final_perturbation_dW"] = str(dW)
        augmented.append(list(rows) + [final_row])
        summary_rows.append({**base_row,
            "x_final": float(x_T), "old_work": float(W_old),
            "dW_final": float(dW), "new_work": float(W_new), "appended": 1,
        })
    return augmented, summary_rows


# ---------------------------------------------------------------------------
# v3-1: NEQ segment runner with augmented protocol files
# ---------------------------------------------------------------------------

def run_neq_segment_v3(
    *, name: str,
    left_boundary: EnsembleWindow, right_boundary: EnsembleWindow,
    left_source: EQCluster | EnsembleWindow, right_source: EQCluster | EnsembleWindow,
    bin_path: str, ctx: dict[str, Any], t_neq: int, n_neq_traj: int,
    seed: int, root: Path, k_min: float, k_max: float,
    neq_pair_source: str = "newly_generated",
) -> NEQSegment:
    """Bidirectional NES with augmented protocol. No NES truncation.

    v3-1: writes protocol_forward_augmented.csv and protocol_reverse_augmented.csv,
    each with an extra endpoint step so augmented trajectory length matches.
    """
    root.mkdir(parents=True, exist_ok=True)
    n_time = max(2, int(t_neq))
    proto = build_linear_bridge_protocol(left_boundary, right_boundary, n_time, k_min, k_max)
    centers_fwd, ks_fwd = proto["centers"], proto["ks"]
    centers_rev, ks_rev = list(reversed(centers_fwd)), list(reversed(ks_fwd))

    # v3-1: augmented protocol = original + endpoint step
    centers_fwd_aug = centers_fwd + [float(right_boundary.center_x)]
    ks_fwd_aug = ks_fwd + [float(right_boundary.k)]
    centers_rev_aug = centers_rev + [float(left_boundary.center_x)]
    ks_rev_aug = ks_rev + [float(left_boundary.k)]

    fwd_path = root / "protocol_forward.csv"
    rev_path = root / "protocol_reverse.csv"
    fwd_path_aug = root / "protocol_forward_augmented.csv"
    rev_path_aug = root / "protocol_reverse_augmented.csv"

    _write_protocol_path(fwd_path, centers_fwd, ks_fwd)
    _write_protocol_path(rev_path, centers_rev, ks_rev)
    _write_protocol_path(fwd_path_aug, centers_fwd_aug, ks_fwd_aug)
    _write_protocol_path(rev_path_aug, centers_rev_aug, ks_rev_aug)

    fwd_root = root / "forward"
    rev_root = root / "reverse"
    fwd_root.mkdir(parents=True, exist_ok=True)
    rev_root.mkdir(parents=True, exist_ok=True)

    neq_nout = max(1, int(math.ceil(float(t_neq) / 100.0)))
    protocol_k_fwd = float(np.mean([abs(k) for k in ks_fwd]))
    k_midscale = float(ctx.get("nes_screen", {}).get("fixed", {}).get("k_midscale", 1.0))

    for direction, eq_left, eq_right, cx_L, cx_R, fpath, out_dir in [
        ("fwd", left_boundary.eq_file, right_boundary.eq_file,
         float(left_boundary.center_x), float(right_boundary.center_x), fwd_path, fwd_root),
        ("rev", right_boundary.eq_file, left_boundary.eq_file,
         float(right_boundary.center_x), float(left_boundary.center_x), rev_path, rev_root),
    ]:
        cmd = [
            bin_path, *build_common_args(ctx),
            "-k", str(protocol_k_fwd), "-k_midscale", str(k_midscale),
            "-A_center", f"{cx_L},0.0", "-B_center", f"{cx_R},0.0",
            "-eq0", str(eq_left), "-eq1", str(eq_right),
            "-fpath", str(fpath), "-N_neq", str(n_neq_traj), "-T_neq", str(t_neq),
            "-neq_nout", str(neq_nout),
            "-neq_seed", str(seed if direction == "fwd" else seed + 1),
            "-out_dir", str(out_dir), "-log", str(out_dir / "neq.log"),
        ]
        run_checked(cmd)

    def read_traj_dir(d: Path) -> tuple[list[list[dict[str, str]]], list[Path]]:
        files = sorted(d.glob("neq_*.csv"))
        return [read_csv_rows(f) for f in files], files

    fwd_trajs_raw, fwd_files = read_traj_dir(fwd_root)
    rev_trajs_raw, rev_files = read_traj_dir(rev_root)

    # Append final perturbation to each trajectory
    fwd_trajs, fwd_perturb = append_final_perturbation(
        fwd_trajs_raw,
        endpoint_center=float(right_boundary.center_x), endpoint_k=float(right_boundary.k),
        last_protocol_center=float(centers_fwd[-1]), last_protocol_k=float(ks_fwd[-1]),
        direction_label="forward",
    )
    rev_trajs, rev_perturb = append_final_perturbation(
        rev_trajs_raw,
        endpoint_center=float(left_boundary.center_x), endpoint_k=float(left_boundary.k),
        last_protocol_center=float(centers_rev[-1]), last_protocol_k=float(ks_rev[-1]),
        direction_label="reverse",
    )

    perturb_rows = fwd_perturb + rev_perturb
    if perturb_rows:
        _perturb_cols = [
            "direction", "traj_index", "x_final", "old_work", "dW_final", "new_work",
            "endpoint_center", "endpoint_k", "last_protocol_center", "last_protocol_k", "appended",
        ]
        write_csv(root / "final_perturbation_summary.csv", _perturb_cols, perturb_rows)

    n_fwd_appended = sum(1 for r in fwd_perturb if r.get("appended", 0))
    n_rev_appended = sum(1 for r in rev_perturb if r.get("appended", 0))

    seg = NEQSegment(
        name=name, left=left_source, right=right_source,
        left_boundary=left_boundary, right_boundary=right_boundary, root=root,
        forward_trajectories=fwd_trajs, reverse_trajectories=rev_trajs,
        forward_trajectory_files=fwd_files, reverse_trajectory_files=rev_files,
        forward_path_file=fwd_path, reverse_path_file=rev_path,
        forward_path_file_augmented=fwd_path_aug,
        reverse_path_file_augmented=rev_path_aug,
        protocol_k=protocol_k_fwd, protocol_mode="linear_v3",
        connectivity={
            "neq_pair_source": neq_pair_source, "final_perturbation_appended": True,
            "n_final_perturbations_forward": n_fwd_appended,
            "n_final_perturbations_reverse": n_rev_appended,
        },
        mts_patch_built=False, cft_summary={},
    )
    write_json(root / "segment_summary.json", {
        "name": name, "left_boundary": left_boundary.name,
        "right_boundary": right_boundary.name, "protocol_mode": "linear_v3",
        "neq_pair_source": neq_pair_source, "final_perturbation_appended": True,
        "n_final_perturbations_forward": n_fwd_appended,
        "n_final_perturbations_reverse": n_rev_appended,
        "n_forward_trajs": len(fwd_trajs), "n_reverse_trajs": len(rev_trajs),
        "protocol_forward_augmented": str(fwd_path_aug),
        "protocol_reverse_augmented": str(rev_path_aug),
    })
    return seg


# ---------------------------------------------------------------------------
# EQ cluster patch (EQ-MBAR)
# ---------------------------------------------------------------------------

def build_eq_cluster_patch_v3(
    cluster: EQCluster, grid: np.ndarray, ctx: dict[str, Any],
    n_boot: int, patch_root: Path, rng_seed: int,
) -> PMFPatch:
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

    write_csv(patch_dir / "pmf.csv", ["x", "pmf", "variance"], [
        {"x": float(grid[i]),
         "pmf": float(base_pmf[i]) if np.isfinite(base_pmf[i]) else "",
         "variance": float(variance[i]) if np.isfinite(variance[i]) else ""}
        for i in range(len(grid))
    ])
    write_json(patch_dir / "patch_summary.json", {
        "name": cluster.name, "kind": "EQ_MBAR", "n_windows": len(cluster.windows),
        "n_boot": n_boot, "left_x": float(cluster.left_x), "right_x": float(cluster.right_x),
    })
    anchor_variances["var_eq_min"] = variance
    return PMFPatch(
        name=cluster.name, kind="EQ_MBAR", root=patch_dir,
        grid=np.asarray(grid, dtype=float), pmf=base_pmf,
        variance=variance, coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[w.name for w in cluster.windows],
        metadata={"cluster_name": cluster.name, "n_boot": n_boot,
                  "left_x": float(cluster.left_x), "right_x": float(cluster.right_x)},
        anchor_variances=anchor_variances,
    )


# ---------------------------------------------------------------------------
# v3-1 + v3-8: NEQ/MTS patch with augmented protocol and proper mts_solved check
# ---------------------------------------------------------------------------

def build_neq_mts_patch_v3(
    segment: NEQSegment, grid: np.ndarray, ctx: dict[str, Any],
    n_boot: int, patch_root: Path, rng_seed: int,
) -> PMFPatch:
    """Build NEQ/MTS patch.

    v3-1: reads augmented protocol file so n_time includes the final perturbation step.
    v3-8: mts_solved checks CFT, finite delta_f, and finite PMF/variance coverage.
    """
    patch_dir = patch_root / segment.name
    patch_dir.mkdir(parents=True, exist_ok=True)

    fwd_frames = [pd.DataFrame(rows) for rows in segment.forward_trajectories]
    rev_frames = [pd.DataFrame(rows) for rows in segment.reverse_trajectories]
    x_fwd, work_fwd = trajectory_frames_to_arrays(fwd_frames)
    x_rev, work_rev = trajectory_frames_to_arrays(rev_frames)

    # v3-1: use augmented protocol (includes endpoint step matching final perturbation row)
    aug_fwd_path = segment.forward_path_file_augmented or segment.forward_path_file
    centers, ks = read_protocol_centers_and_k(aug_fwd_path)

    n_time_protocol_aug = len(centers)
    n_time_work = min(
        x_fwd.shape[1] if x_fwd.ndim == 2 else 0,
        work_fwd.shape[1] if work_fwd.ndim == 2 else 0,
        x_rev.shape[1] if x_rev.ndim == 2 else 0,
        work_rev.shape[1] if work_rev.ndim == 2 else 0,
    )
    n_time = min(n_time_protocol_aug, n_time_work)
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
    cft = solve_segment_cft_delta_f_once(
        work_fwd[:, :n_time], work_rev[:, :n_time], kT=kT
    )
    cft_solved = bool(cft.get("cft_solved", False))
    finite_delta_f = cft_solved and cft.get("delta_f") is not None and math.isfinite(float(cft["delta_f"]))
    fixed_delta_f = float(cft["delta_f"]) if finite_delta_f else None

    pmf_arr = np.full(len(grid), np.nan, dtype=float)
    var_arr = np.full(len(grid), np.nan, dtype=float)
    boot_n_used = 0
    mts_exception = ""

    try:
        left_ref_x = float(segment.left_boundary.mean_x)
        boot_result = bootstrap_bidirectional_mts_pmf(
            x_fwd[:, :n_time], work_fwd[:, :n_time],
            x_rev[:, :n_time], work_rev[:, :n_time],
            centers, ks, grid,
            reference_x=left_ref_x, kT=kT,
            n_boot=int(n_boot), fk_boot=max(int(n_boot // 8), 4),
            rng_seed=int(rng_seed),
            fixed_delta_f=fixed_delta_f,
            recompute_delta_f_per_bootstrap=(fixed_delta_f is None),
        )
        pmf_arr, var_arr = get_bootstrap_pmf_and_variance(boot_result)
        boot_n_used = int(boot_result.get("n_boot_used", 0))
    except Exception as exc:
        mts_exception = str(exc)

    finite_pmf = np.any(np.isfinite(pmf_arr))
    finite_var = np.any(np.isfinite(var_arr))

    # v3-8: proper mts_solved check
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
        "final_perturbation_appended": True,
        "final_perturbation_used_in_cft": True,  # v3-1 diagnostic
        "final_perturbation_used_in_mts": True,  # v3-1 diagnostic
        "n_time_protocol_augmented": n_time_protocol_aug,
        "n_time_work_array": n_time_work,
        "n_boot_used": boot_n_used,
    }
    segment.mts_patch_built = mts_solved

    write_json(patch_dir / "patch_summary.json", {
        "name": segment.name, "kind": "NEQ_MTS",
        "cft_solved": cft_solved, "mts_solved": mts_solved,
        "n_time_protocol_augmented": n_time_protocol_aug,
        "n_time_work_array": n_time_work,
        "delta_f": cft.get("delta_f"), "n_boot": n_boot,
        "final_perturbation_used_in_cft": True,
        "final_perturbation_used_in_mts": True,
    })
    write_csv(patch_dir / "pmf.csv", ["x", "pmf", "variance"], [
        {"x": float(grid[i]),
         "pmf": float(pmf_arr[i]) if np.isfinite(pmf_arr[i]) else "",
         "variance": float(var_arr[i]) if np.isfinite(var_arr[i]) else ""}
        for i in range(len(grid))
    ])

    return PMFPatch(
        name=segment.name, kind="NEQ_MTS", root=patch_dir,
        grid=np.asarray(grid, dtype=float), pmf=pmf_arr,
        variance=var_arr, coverage_mask=np.asarray(coverage, dtype=bool),
        source_names=[segment.left_boundary.name, segment.right_boundary.name],
        metadata={"segment_name": segment.name, "cft_solved": cft_solved,
                  "mts_solved": mts_solved, "delta_f": cft.get("delta_f"),
                  "n_boot": n_boot, "final_perturbation_appended": True},
    )


# ---------------------------------------------------------------------------
# Global PMF fusion: inverse-precision variance
# ---------------------------------------------------------------------------

def fit_global_pmf_v3(
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


def write_state_tables_v3(
    base_root: Path, windows: list[EnsembleWindow], clusters: list[EQCluster],
    segments: list[NEQSegment], patches: list[PMFPatch],
    fit_details: dict[str, Any], overlap_rows: list[dict[str, Any]],
    grid: np.ndarray, global_pmf: np.ndarray, global_var: np.ndarray,
) -> None:
    base_root.mkdir(parents=True, exist_ok=True)

    # v3-7: x_most removed from windows.csv
    write_csv(base_root / "windows.csv", [
        "name", "center_x", "k", "mean_x", "std_x", "generation", "side",
    ], [
        {"name": w.name, "center_x": w.center_x, "k": w.k, "mean_x": w.mean_x,
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
# v3-9 + v3-3 + v3-6: Adaptive connectivity refinement (one pair per round)
# ---------------------------------------------------------------------------

_REFINEMENT_COLS = [
    "round", "target_pair", "left_window", "right_window",
    "left_mean_x", "right_mean_x", "segment_type", "nes_action",
    "cft_solved", "mts_patch_built", "mts_failure_reason",
    "fallback_used", "fallback_rule",
    "inserted_window", "inserted_center_x", "inserted_k",
    "target_mean", "target_sigma", "k_raw", "k_clipped",
    "reason", "n_clusters_before", "n_clusters_after", "used_steps",
]


def _find_existing_segment(
    segments: list[NEQSegment], left_name: str, right_name: str
) -> NEQSegment | None:
    for seg in segments:
        if seg.left_boundary.name == left_name and seg.right_boundary.name == right_name:
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


def run_connectivity_refinement_v3(
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

    v3-9: rebuilds clusters from scratch each round; never iterates over stale list.
    v3-3: basin-like → midpoint EQ only (no NES).
         transition-like → bidirectional NES/MTS (v3-6 fallback on MTS failure).
    v3-4: BAR failure means disconnected; raw sigmoid not used.
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
        # v3-9: rebuild clusters from scratch each round
        clusters, overlap_rows = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))
        if eq_network_is_connected(clusters):
            break

        # v3-9: get first disconnected pair from fresh clusters
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

        # v3-3 + v3-5 (refinement): classify using boundary window means
        is_transition, bg = classify_segment(left_boundary, right_boundary)
        row["segment_type"] = "barrier_like" if is_transition else "basin_like"

        if not is_transition:
            # v3-3 basin: no NES — insert midpoint EQ window
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
            new_win = run_eq_window_v3(
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
            # v3-3 transition: bidirectional NES/MTS
            existing_seg = _find_existing_segment(segments, left_boundary.name, right_boundary.name)

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
                seg = run_neq_segment_v3(
                    name=seg_name, left_boundary=left_boundary, right_boundary=right_boundary,
                    left_source=left_cluster, right_source=right_cluster,
                    bin_path=bin_path, ctx=ctx, t_neq=args.t_neq, n_neq_traj=args.n_neq_traj,
                    seed=int(args.seed) + 500000 + round_idx * 100,
                    root=seg_root, k_min=float(args.k_min), k_max=float(args.k_max),
                    neq_pair_source="newly_generated",
                )
                segments.append(seg)
                row["nes_action"] = "newly_generated"

            # Build MTS patch
            patch_root = out_root / "refinement" / f"round_{round_idx:03d}" / "patches"
            mts_patch = build_neq_mts_patch_v3(
                seg, grid, ctx, int(args.n_bootstrap_neq), patch_root,
                rng_seed=int(args.seed) + 600000 + round_idx * 100,
            )
            cft_solved = bool(seg.cft_summary.get("cft_solved", False))
            mts_solved = bool(seg.cft_summary.get("mts_solved", False))
            row["cft_solved"] = int(cft_solved)
            row["mts_patch_built"] = int(mts_solved)

            if mts_solved and np.any(np.isfinite(mts_patch.pmf)) and np.any(np.isfinite(mts_patch.variance)):
                # MTS succeeded — keep patch; no EQ insertion needed this round
                row["reason"] = "transition_mts_patch_built"
            else:
                # v3-6: MTS failed — fallback to midpoint mean-only GT EQ insertion
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
                new_win = run_eq_window_v3(
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

        # v3-9: rebuild clusters after each insertion/repair
        clusters, overlap_rows = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))
        row["n_clusters_after"] = len(clusters)

        # Rebuild EQ patches and provisional PMF
        eq_patches: list[PMFPatch] = []
        for ci, cl in enumerate(clusters):
            eq_patches.append(build_eq_cluster_patch_v3(
                cl, grid, ctx, int(args.n_bootstrap_eq),
                out_root / "refinement" / f"round_{round_idx:03d}" / "patches" / "eq",
                rng_seed=int(args.seed) + 400000 + round_idx * 1000 + ci,
            ))

        if eq_network_is_connected(clusters):
            patches_for_global = eq_patches
            patch_selection_rule = "connected_EQ_MBAR_only"
        else:
            # Only include MTS patches that actually solved (v3-8)
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

        global_pmf, global_var, fit_details = fit_global_pmf_v3(
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


def run_final_eq_extension_v3(
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
            ext_rows_csv = read_csv_rows(ext_win_root / "eq_window.csv")
            ext_tail = tail_rows_from_eq_rows(ext_rows_csv, float(args.tail_fraction))
            window.tail_rows = list(window.tail_rows) + ext_tail
            tail_x = np.asarray(
                [float(r["x"]) for r in window.tail_rows if r.get("x", "") != ""], dtype=float
            )
            tail_x = tail_x[np.isfinite(tail_x)]
            if tail_x.size >= 2:
                window.mean_x = float(np.mean(tail_x))
                window.std_x = float(np.std(tail_x, ddof=1))

        budget.spend(round_cost, f"final_eq_extension_round_{round_index:03d}",
                     "EQ_EXTENSION", "final_eq_extension")

        new_clusters, new_overlap = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))
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
            new_patches.append(build_eq_cluster_patch_v3(
                cl, grid, ctx, int(args.n_bootstrap_eq),
                round_root / "patches",
                rng_seed=base_seed + 300000 + round_index * 1000 + ci,
            ))
        global_pmf, global_var, fit_details = fit_global_pmf_v3(
            new_patches, grid, float(args.variance_floor)
        )
        patches_for_global = new_patches
        fit_details["patch_selection_rule"] = "connected_EQ_MBAR_only"
        fit_details["eq_network_connected"] = True
        fit_details["final_estimator"] = "connected_EQ_MBAR_only"

        write_state_tables_v3(round_root / "state", windows, clusters, segments,
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
        description="MiNES v3 — augmented protocol, strict BAR connectivity, "
                    "NES-only for transitions, MTS failure fallback, no x_most."
    )
    parser.add_argument("--system-root", required=True)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--label", default="mines_variance_fusion_v3")
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

    # -----------------------------------------------------------------------
    # Stage 0: Sample L0 and R0 endpoints
    # -----------------------------------------------------------------------
    for _ in range(2):
        if not budget.can_spend(eq_cost(args.n_eq_steps)):
            raise RuntimeError("Budget too small for endpoint EQ sampling.")
        budget.spend(eq_cost(args.n_eq_steps), "endpoint", "EQ", "init")

    left0 = run_eq_window_v3(
        name="L_gen0", center_x=float(ctx["mines_screen"]["fixed"]["start_x_left"]),
        k=float(args.k_min), generation=0, side="left",
        bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
        eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
        seed=seed + 1, root=out_root / "windows" / "L_gen0",
    )
    right0 = run_eq_window_v3(
        name="R_gen0", center_x=float(ctx["mines_screen"]["fixed"]["start_x_right"]),
        k=float(args.k_min), generation=0, side="right",
        bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
        eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
        seed=seed + 2, root=out_root / "windows" / "R_gen0",
    )
    windows.extend([left0, right0])
    profile = make_global_gt_profile(left0, right0)
    left_frontier, right_frontier = left0, right0

    clusters, overlap_rows = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))
    patches_for_global: list[PMFPatch] = []
    global_pmf = np.full(len(grid), np.nan, dtype=float)
    global_var = np.full(len(grid), np.nan, dtype=float)
    fit_details: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Stage 1: Exploratory chain growth
    # -----------------------------------------------------------------------
    for generation in range(int(args.max_generations)):
        clusters, overlap_rows = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))
        if eq_network_is_connected(clusters):
            print(f"[v3] EQ network connected at generation {generation}. Stopping exploration.")
            break

        seg_name = f"seg_L{left_frontier.name}_R{right_frontier.name}_gen{generation}"
        n_cost = neq_cost(args.n_neq_traj, args.t_neq)
        if not budget.can_spend(n_cost):
            print("[v3] Budget exhausted before NEQ at generation", generation)
            break
        budget.spend(n_cost, seg_name, "NEQ", f"generation_{generation}")
        seg = run_neq_segment_v3(
            name=seg_name, left_boundary=left_frontier, right_boundary=right_frontier,
            left_source=left_frontier, right_source=right_frontier,
            bin_path=bin_path, ctx=ctx, t_neq=args.t_neq, n_neq_traj=args.n_neq_traj,
            seed=seed + 100 + generation,
            root=out_root / "segments" / seg_name,
            k_min=float(args.k_min), k_max=float(args.k_max),
        )
        segments.append(seg)

        # Diagnostic MTS patch for the frontier segment
        mts_patch = build_neq_mts_patch_v3(
            seg, grid, ctx, int(args.n_bootstrap_neq),
            out_root / "patches" / f"gen{generation}",
            rng_seed=seed + 200 + generation,
        )
        # Store patch ref on segment for later PMF routing
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
            print("[v3] Budget exhausted before EQ children at generation", generation)
            break
        budget.spend(2 * eq_cost(args.n_eq_steps), f"children_gen{generation}",
                     "EQ", f"generation_{generation}")

        left_name = f"L_gen{generation + 1}"
        right_name = f"R_gen{generation + 1}"

        left_child = run_eq_window_v3(
            name=left_name, center_x=float(left_design["x_child"]),
            k=float(left_design["k_child"]), generation=generation + 1, side="left",
            bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
            eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
            seed=seed + 300 + generation * 2, root=out_root / "windows" / left_name,
        )
        windows.append(left_child)
        # v3-5: rebuild after left child
        clusters, overlap_rows = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))

        right_child = run_eq_window_v3(
            name=right_name, center_x=float(right_design["x_child"]),
            k=float(right_design["k_child"]), generation=generation + 1, side="right",
            bin_path=bin_path, ctx=ctx, n_eq_steps=args.n_eq_steps,
            eq_save_every=args.eq_save_every, tail_fraction=args.tail_fraction,
            seed=seed + 300 + generation * 2 + 1, root=out_root / "windows" / right_name,
        )
        windows.append(right_child)
        # v3-5: rebuild after right child
        clusters, overlap_rows = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))

        left_frontier = left_child
        right_frontier = right_child
        left_design["generation"] = generation
        right_design["generation"] = generation
        generation_rows.extend([left_design, right_design])

        # Build EQ patches and provisional PMF
        eq_patches: list[PMFPatch] = []
        for ci, cl in enumerate(clusters):
            eq_patches.append(build_eq_cluster_patch_v3(
                cl, grid, ctx, int(args.n_bootstrap_eq),
                out_root / "patches" / f"gen{generation}" / "eq",
                rng_seed=seed + 400 + generation * 100 + ci,
            ))

        if eq_network_is_connected(clusters):
            patches_for_global = eq_patches
            patch_selection_rule = "connected_EQ_MBAR_only"
        else:
            # v3-8: only include mts_solved patches
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

        global_pmf, global_var, fit_details = fit_global_pmf_v3(
            patches_for_global, grid, float(args.variance_floor)
        )
        fit_details["patch_selection_rule"] = patch_selection_rule
        fit_details["eq_network_connected"] = bool(eq_network_is_connected(clusters))
        fit_details["final_estimator"] = (
            "connected_EQ_MBAR_only" if eq_network_is_connected(clusters) else "provisional_fused_pmf"
        )

        gen_root = out_root / f"generation_{generation:03d}"
        write_state_tables_v3(gen_root, windows, clusters, segments,
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
    clusters, overlap_rows = build_eq_clusters_v3(windows, ctx, float(args.eq_overlap_threshold))

    (windows, clusters, segments, patches_for_global,
     global_pmf, global_var, fit_details, overlap_rows, ref_rows) = run_connectivity_refinement_v3(
        windows=windows, clusters=clusters, segments=segments,
        patches_for_global=patches_for_global, global_pmf=global_pmf, global_var=global_var,
        fit_details=fit_details, overlap_rows=overlap_rows, args=args, ctx=ctx, grid=grid,
        out_root=out_root, bin_path=bin_path, budget=budget, quality_rows=quality_rows,
        profile=profile, beta_eff=beta_eff,
    )

    write_state_tables_v3(out_root, windows, clusters, segments,
                           patches_for_global, fit_details, overlap_rows,
                           grid, global_pmf, global_var)

    # -----------------------------------------------------------------------
    # Stage 3: Final connected-EQ MBAR extension
    # -----------------------------------------------------------------------
    clusters, patches_for_global, global_pmf, global_var, fit_details, overlap_rows, ext_rows = \
        run_final_eq_extension_v3(
            windows=windows, clusters=clusters, segments=segments,
            patches_for_global=patches_for_global,
            global_pmf=global_pmf, global_var=global_var, fit_details=fit_details,
            overlap_rows=overlap_rows, args=args, ctx=ctx, grid=grid,
            out_root=out_root, bin_path=bin_path, budget=budget, quality_rows=quality_rows,
        )

    # -----------------------------------------------------------------------
    # Final outputs
    # -----------------------------------------------------------------------
    write_state_tables_v3(out_root, windows, clusters, segments,
                           patches_for_global, fit_details, overlap_rows,
                           grid, global_pmf, global_var)

    if quality_rows:
        write_csv(out_root / "pmf_quality_vs_steps.csv",
                  ordered_fieldnames(quality_rows), quality_rows)
    budget.write(out_root / "budget_ledger.csv")

    ext_stop = ext_rows[-1].get("stop_reason", "") if ext_rows else "no_extension"
    ref_stop = ref_rows[-1].get("reason", "") if ref_rows else "no_refinement"
    summary = {
        "label": label, "seed": seed,
        "n_windows": len(windows), "n_segments": len(segments), "n_clusters": len(clusters),
        "eq_network_connected": bool(eq_network_is_connected(clusters)),
        "final_estimator": fit_details.get("final_estimator", ""),
        "patch_selection_rule": fit_details.get("patch_selection_rule", ""),
        "final_extension_stop_reason": ext_stop,
        "refinement_stop_reason": ref_stop,
        "used_steps": int(budget.used_steps),
        "total_budget_steps": int(budget.total_budget_steps),
        "protocol": "mines_variance_fusion_v3",
        "eq_overlap_threshold": float(args.eq_overlap_threshold),
        "target_kl": float(args.target_kl),
        "overlap_method": "bar_offset_corrected_strict",
        "variance_formula": "inverse_precision",
        "final_perturbation_in_cft_mts": True,
        "x_most_removed": True,
    }
    write_json(out_root / "mines_variance_fusion_summary.json", summary)
    print(str(out_root / "mines_variance_fusion_summary.json"))


if __name__ == "__main__":
    main()
