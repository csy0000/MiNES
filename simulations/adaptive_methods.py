#!/usr/bin/env python3
"""Launch the raw adaptive benchmark simulations for AUS and MINES.

This module is intentionally thin: it orchestrates calls into the compiled
`neq_sim` binary, writes lightweight raw-method summaries, and leaves PMF
reconstruction to the Python analysis reducer.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.interpolate import CubicSpline


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "src" / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from bidirectional_mts_pmf import (  # noqa: E402
    align_pmf_to_reference,
    align_pmf_to_endpoint_average_zero,
    bootstrap_bidirectional_mts_pmf,
    build_bidirectional_mts_pmf,
    estimate_intermediate_reduced_free_energies,
)


_PYMBAR = None


def load_pymbar():
    global _PYMBAR
    if _PYMBAR is None:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                import pymbar as pymbar_module
        except ImportError as exc:  # pragma: no cover - environment/config issue
            raise RuntimeError(
                "pymbar is required for adaptive MBAR reconstruction. "
                "Install it in the active Python environment before running adaptive methods."
            ) from exc
        _PYMBAR = pymbar_module
    return _PYMBAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    aus = subparsers.add_parser("run-aus-seed")
    aus.add_argument("--system-root", required=True)
    aus.add_argument("--combo-label", required=True)
    aus.add_argument("--seed", type=int, required=True)
    aus.add_argument("--bin", required=True)

    mines = subparsers.add_parser("run-mines-seed")
    mines.add_argument("--system-root", required=True)
    mines.add_argument("--combo-label", required=True)
    mines.add_argument("--seed", type=int, required=True)
    mines.add_argument("--bin", required=True)

    mines_current = subparsers.add_parser("run-mines-current-protocol")
    mines_current.add_argument("--system-root", required=True)
    mines_current.add_argument("--seed", type=int, required=True)
    mines_current.add_argument("--bin", required=True)
    mines_current.add_argument("--label", default="current_protocol")
    mines_current.add_argument("--t-neq", type=int, default=None)
    mines_current.add_argument("--n-neq-traj", type=int, default=None)
    mines_current.add_argument("--total-budget-steps", type=int, default=None)

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def slug_float(value: float) -> str:
    text = f"{value}"
    return text.replace("-", "m").replace(".", "p")


def build_grid(xmin: float, xmax: float, dx: float) -> np.ndarray:
    n = int(round((xmax - xmin) / dx))
    return np.asarray([xmin + dx * i for i in range(n + 1)], dtype=float)


def grid_index(value: float, grid: np.ndarray) -> int | None:
    # Continuous trajectories are assigned to the nearest uniform-grid point
    # rather than requiring exact float equality with a grid coordinate.
    if len(grid) == 0:
        return None
    if len(grid) == 1:
        return 0 if abs(float(value) - float(grid[0])) <= 1.0e-9 else None
    dx = float(grid[1] - grid[0])
    idx = int(round((float(value) - float(grid[0])) / dx))
    if idx < 0 or idx >= len(grid):
        return None
    if abs(float(grid[idx]) - float(value)) > 0.5 * abs(dx) + 1.0e-9:
        return None
    return idx


def histogram_counts(values: list[float], grid: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(grid), dtype=float)
    for value in values:
        idx = grid_index(float(value), grid)
        if idx is not None:
            counts[idx] += 1.0
    return counts


def histogram_prob(values: list[float], grid: np.ndarray) -> np.ndarray:
    # The adaptive methods reason on the exact analysis grid rather than on a
    # smoothed KDE so that their placement rules stay easy to audit.
    if not values:
        return np.zeros(len(grid), dtype=float)
    counts = histogram_counts(values, grid)
    total = float(np.sum(counts))
    if total > 0.0:
        counts /= total
    return counts


def tail_fraction_values(values: list[float], keep_fraction: float) -> list[float]:
    """Keep only the trailing fraction of an ordered trajectory sample list."""
    if not values:
        return []
    keep_fraction = float(keep_fraction)
    if keep_fraction >= 1.0:
        return list(values)
    if keep_fraction <= 0.0:
        return [float(values[-1])]
    discard_count = int(math.floor(len(values) * (1.0 - keep_fraction)))
    if discard_count >= len(values):
        discard_count = len(values) - 1
    return [float(value) for value in values[discard_count:]]


def downsample_ordered_values(values: list[float], max_points: int) -> list[float]:
    """Uniformly subsample an ordered trajectory while keeping endpoints."""
    if max_points <= 0 or len(values) <= max_points:
        return list(values)
    indices = np.linspace(0, len(values) - 1, num=max_points, dtype=int)
    return [float(values[int(idx)]) for idx in indices]


def overlap_coefficient(values_a: list[float], values_b: list[float], grid: np.ndarray) -> float:
    prob_a = histogram_prob(values_a, grid)
    prob_b = histogram_prob(values_b, grid)
    return float(np.sum(np.minimum(prob_a, prob_b)))


def js_divergence(values_a: list[float], values_b: list[float], grid: np.ndarray) -> float:
    """Discrete Jensen-Shannon divergence on the shared analysis grid.

    The implementation uses base-2 logarithms so the divergence is bounded by
    `1.0`; the active current-protocol overlap gate compares the raw
    divergence directly against `0.8`.
    """

    prob_a = histogram_prob(values_a, grid)
    prob_b = histogram_prob(values_b, grid)
    midpoint = 0.5 * (prob_a + prob_b)
    positive_a = prob_a > 0.0
    positive_b = prob_b > 0.0
    kl_a = (
        float(np.sum(prob_a[positive_a] * np.log2(prob_a[positive_a] / midpoint[positive_a])))
        if np.any(positive_a)
        else 0.0
    )
    kl_b = (
        float(np.sum(prob_b[positive_b] * np.log2(prob_b[positive_b] / midpoint[positive_b])))
        if np.any(positive_b)
        else 0.0
    )
    return 0.5 * (kl_a + kl_b)


def reliable_reach(values: list[float], grid: np.ndarray, ess_min: float, side: str, current_center: float) -> float | None:
    # A frontier can only advance to grid points that have enough effective
    # sample support in the current restrained window.
    if not values:
        return None
    counts = histogram_counts(values, grid)
    if side == "left":
        candidates = [float(x) for x, c in zip(grid, counts) if c >= ess_min and x > current_center + 1.0e-9]
        return max(candidates) if candidates else None
    candidates = [float(x) for x, c in zip(grid, counts) if c >= ess_min and x < current_center - 1.0e-9]
    return min(candidates) if candidates else None


def local_pmf_from_samples(values: list[float], grid: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    counts = histogram_counts(values, grid)
    pmf = np.full(len(grid), np.inf, dtype=float)
    resolved = counts > 0.0
    if np.any(resolved):
        probs = counts[resolved] / float(np.sum(counts[resolved]))
        pmf[resolved] = -(1.0 / beta) * np.log(probs)
        pmf[resolved] -= float(np.min(pmf[resolved]))
    return counts, pmf


def parent_sigma(values: list[float]) -> float:
    if not values:
        return 0.0
    sigma = float(np.std(np.asarray(values, dtype=float)))
    return sigma if math.isfinite(sigma) else 0.0


def parent_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = float(np.mean(np.asarray(values, dtype=float)))
    return mu if math.isfinite(mu) else 0.0


def nearest_grid_value(value: float, grid: np.ndarray) -> float:
    if len(grid) == 0:
        return float(value)
    idx = int(np.argmin(np.abs(grid - float(value))))
    return float(grid[idx])


def quantile_band(values: list[float], q_next_level: float) -> tuple[float, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    lo_level = min(float(q_next_level), 1.0 - float(q_next_level))
    hi_level = max(float(q_next_level), 1.0 - float(q_next_level))
    return float(np.quantile(arr, lo_level)), float(np.quantile(arr, hi_level))


def logsumexp_np(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(max_values), max_values, 0.0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        summed = np.sum(np.exp(values - safe_max), axis=axis, keepdims=True)
        result = safe_max + np.log(summed)
    result = np.where(np.isfinite(max_values), result, -np.inf)
    if axis is None:
        return np.asarray(result).reshape(())
    return np.squeeze(result, axis=axis)


def fill_nan_with_slope(free_energy: np.ndarray, grid: np.ndarray, slope: float) -> np.ndarray:
    """Fill unresolved PMF bins with a monotone edge extension.

    Internal gaps are linearly interpolated between resolved bins. Unresolved
    bins beyond the resolved PMF range are extended outward with a fixed
    positive slope so the smoothing pass has a stable background to work with.
    """
    arr = np.asarray(free_energy, dtype=float).copy()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    finite_idx = np.flatnonzero(finite)
    if finite_idx.size == len(arr):
        return arr

    first = int(finite_idx[0])
    last = int(finite_idx[-1])
    arr[~finite] = np.interp(grid[~finite], grid[finite], arr[finite])

    slope = abs(float(slope))
    if first > 0:
        first_x = float(grid[first])
        first_f = float(arr[first])
        arr[:first] = first_f + slope * (first_x - grid[:first])
    if last < len(arr) - 1:
        last_x = float(grid[last])
        last_f = float(arr[last])
        arr[last + 1 :] = last_f + slope * (grid[last + 1 :] - last_x)
    return arr


def reduced_doublewell_potential_np(values: np.ndarray, ctx: dict) -> np.ndarray:
    pot = ctx["potential"]
    beta = 1.0 / float(ctx["thermal_kT"])
    u0 = pot["k0"] * (values - pot["x0"]) * (values - pot["x0"])
    u1 = pot["k1"] * (values - pot["x1"]) * (values - pot["x1"])
    log_t0 = -beta * u0
    log_t1 = -beta * u1 - pot["E1"]
    log_max = np.maximum(log_t0, log_t1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return -(log_max + np.log(np.exp(log_t0 - log_max) + np.exp(log_t1 - log_max)))


def us_mbar_profile(
    window_samples: list[np.ndarray],
    window_centers: list[float],
    window_ks: list[float],
    grid: np.ndarray,
    ctx: dict,
    initial_f: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    free_energy, reduced_free, _, _ = us_mbar_profile_details(
        window_samples,
        window_centers,
        window_ks,
        grid,
        ctx,
        initial_f,
    )
    return free_energy, reduced_free


def us_mbar_profile_details(
    window_samples: list[np.ndarray],
    window_centers: list[float],
    window_ks: list[float],
    grid: np.ndarray,
    ctx: dict,
    initial_f: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Reuse the same PyMBAR-based estimator as the benchmark reducer so
    # adaptive placement decisions and final analysis are based on the same PMF
    # reconstruction rule.
    nonempty = [
        (samples, center, k)
        for samples, center, k in zip(window_samples, window_centers, window_ks)
        if samples.size > 0
    ]
    if not nonempty:
        return (
            np.full(len(grid), np.nan, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(len(grid), dtype=float),
            np.zeros(len(grid), dtype=float),
        )

    sample_arrays = [item[0] for item in nonempty]
    centers = np.asarray([item[1] for item in nonempty], dtype=float)
    ks = np.asarray([item[2] for item in nonempty], dtype=float)
    counts = np.asarray([arr.size for arr in sample_arrays], dtype=float)
    samples = np.concatenate(sample_arrays)

    beta = 1.0 / float(ctx["thermal_kT"])
    reduced_unbiased = reduced_doublewell_potential_np(samples, ctx)
    reduced_biased = reduced_unbiased[None, :] + 0.5 * beta * ks[:, None] * (samples[None, :] - centers[:, None]) ** 2

    pymbar = load_pymbar()
    u_kn = np.vstack([reduced_biased, reduced_unbiased[None, :]])
    N_k = np.concatenate([counts.astype(int), np.asarray([0], dtype=int)])
    initial_f_k = None
    if initial_f is not None and initial_f.size == counts.size:
        initial_f_k = np.zeros(len(N_k), dtype=float)
        initial_f_k[:-1] = np.asarray(initial_f, dtype=float)
    mbar = pymbar.MBAR(
        u_kn,
        N_k,
        maximum_iterations=200,
        relative_tolerance=1.0e-6,
        verbose=False,
        initial_f_k=initial_f_k,
        initialize="zeros",
    )
    reduced_free = np.asarray(mbar.f_k[:-1], dtype=float)
    if reduced_free.size > 0:
        reduced_free -= reduced_free[0]
    weights = np.asarray(mbar.W_nk[:, -1], dtype=float)

    if len(grid) > 1:
        dx = float(grid[1] - grid[0])
    else:
        dx = 1.0
    xmin = float(grid[0])
    indices = np.floor((samples - xmin) / dx + 0.5).astype(int)
    valid = (indices >= 0) & (indices < len(grid))
    probability = np.zeros(len(grid), dtype=float)
    np.add.at(probability, indices[valid], weights[valid])
    weight_squares = np.zeros(len(grid), dtype=float)
    np.add.at(weight_squares, indices[valid], weights[valid] * weights[valid])
    ess = np.zeros(len(grid), dtype=float)
    positive_weight_sq = weight_squares > 0.0
    ess[positive_weight_sq] = (probability[positive_weight_sq] * probability[positive_weight_sq]) / weight_squares[positive_weight_sq]

    free_energy = np.full(len(grid), np.nan, dtype=float)
    positive = probability > 0.0
    free_energy[positive] = -float(ctx["thermal_kT"]) * np.log(probability[positive])
    finite = np.isfinite(free_energy)
    if np.any(finite):
        free_energy[finite] -= np.min(free_energy[finite])
    return free_energy, reduced_free, probability, ess


def local_fit_window(
    grid: np.ndarray,
    free_energy: np.ndarray,
    probe_x: float,
    window_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.flatnonzero(np.isfinite(free_energy))
    if finite.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    ordered = sorted(
        (int(idx) for idx in finite),
        key=lambda idx: abs(float(grid[idx]) - float(probe_x)),
    )
    keep = ordered[: max(int(window_points), 3)]
    keep = sorted(set(keep))
    x_fit = np.asarray([float(grid[idx]) for idx in keep], dtype=float)
    y_fit = np.asarray([float(free_energy[idx]) for idx in keep], dtype=float)
    return x_fit, y_fit


def evaluate_local_fit(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    probe_x: float,
    fit_method: str,
) -> tuple[float, np.ndarray | None]:
    if x_fit.size < 3 or np.unique(x_fit).size < 3:
        return 0.0, None

    method = str(fit_method)
    probe_x = float(probe_x)
    try:
        if method == "poly_4term_parent":
            if x_fit.size < 4 or np.unique(x_fit).size < 4:
                return 0.0, None
            local_x = x_fit - probe_x
            coeffs = np.polyfit(local_x, y_fit, deg=3)
            slope = float(coeffs[2])
            return (slope if math.isfinite(slope) else 0.0), np.asarray(np.polyval(coeffs, local_x), dtype=float)

        if method == "cubic_spline_parent":
            order = np.argsort(x_fit)
            x_sorted = np.asarray(x_fit[order], dtype=float)
            y_sorted = np.asarray(y_fit[order], dtype=float)
            if x_sorted.size < 3 or np.unique(x_sorted).size < 3:
                return 0.0, None
            spline = CubicSpline(x_sorted, y_sorted, bc_type="natural")
            slope = float(spline(probe_x, 1))
            return (slope if math.isfinite(slope) else 0.0), np.asarray(spline(x_fit), dtype=float)

        if method == "chebyshev_3term":
            domain = [float(np.min(x_fit)), float(np.max(x_fit))]
            if abs(domain[1] - domain[0]) < 1.0e-12:
                return 0.0, None
            model = Chebyshev.fit(x_fit, y_fit, deg=2, domain=domain)
            slope = float(model.deriv()(probe_x))
            return (slope if math.isfinite(slope) else 0.0), np.asarray(model(x_fit), dtype=float)

        local_x = x_fit - probe_x
        coeffs = np.polyfit(local_x, y_fit, deg=2)
        slope = float(coeffs[1])
        return (slope if math.isfinite(slope) else 0.0), np.asarray(np.polyval(coeffs, local_x), dtype=float)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return 0.0, None


def estimate_profile_slope(
    grid: np.ndarray,
    free_energy: np.ndarray,
    probe_x: float,
    fit_method: str,
    window_points: int,
) -> float:
    x_fit, y_fit = local_fit_window(grid, free_energy, probe_x, window_points)
    slope, _ = evaluate_local_fit(x_fit, y_fit, probe_x, fit_method)
    return float(slope) if math.isfinite(slope) else 0.0


def estimate_parent_window_slope(
    parent_values: np.ndarray,
    parent_center: float,
    parent_k: float,
    grid: np.ndarray,
    ctx: dict,
    probe_x: float,
    fit_method: str,
) -> float:
    if parent_values.size == 0:
        return 0.0
    parent_pmf, _ = us_mbar_profile(
        [np.asarray(parent_values, dtype=float)],
        [float(parent_center)],
        [float(parent_k)],
        grid,
        ctx,
        None,
    )
    finite = np.flatnonzero(np.isfinite(parent_pmf))
    if finite.size < 4:
        return 0.0
    x_fit = np.asarray([float(grid[idx]) for idx in finite], dtype=float)
    y_fit = np.asarray([float(parent_pmf[idx]) for idx in finite], dtype=float)
    slope, _ = evaluate_local_fit(x_fit, y_fit, probe_x, fit_method)
    return float(slope) if math.isfinite(slope) else 0.0


def interested_region_ess_indices(
    grid: np.ndarray,
    free_energy: np.ndarray,
    ess: np.ndarray,
    region_left_x: float,
    region_right_x: float,
    half_width: int,
    excluded_indices: set[int] | None = None,
) -> list[int]:
    lo = min(float(region_left_x), float(region_right_x))
    hi = max(float(region_left_x), float(region_right_x))
    excluded = excluded_indices or set()
    candidates: list[int] = []
    for idx in range(len(grid)):
        if idx in excluded:
            continue
        if not np.isfinite(free_energy[idx]) or not math.isfinite(float(ess[idx])) or float(ess[idx]) <= 0.0:
            continue
        x = float(grid[idx])
        if x < lo or x > hi:
            continue
        start = max(0, idx - half_width)
        stop = min(len(grid), idx + half_width + 1)
        if int(np.count_nonzero(np.isfinite(free_energy[start:stop]))) < 4:
            continue
        candidates.append(idx)
    return candidates


def interested_region_ess_state(
    grid: np.ndarray,
    free_energy: np.ndarray,
    ess: np.ndarray,
    region_left_x: float,
    region_right_x: float,
    half_width: int,
    min_fraction: float,
    excluded_indices: set[int] | None = None,
) -> dict | None:
    candidates = interested_region_ess_indices(
        grid,
        free_energy,
        ess,
        region_left_x,
        region_right_x,
        half_width,
        excluded_indices=excluded_indices,
    )
    if not candidates:
        return None
    ess_values = np.asarray([float(ess[idx]) for idx in candidates], dtype=float)
    average_ess = float(np.mean(ess_values))
    lo = min(float(region_left_x), float(region_right_x))
    hi = max(float(region_left_x), float(region_right_x))
    midpoint = 0.5 * (lo + hi)
    low_idx = min(
        candidates,
        key=lambda idx: (float(ess[idx]), abs(float(grid[idx]) - midpoint)),
    )
    threshold = float(min_fraction) * average_ess
    low_ess_value = float(ess[low_idx])
    return {
        "candidate_indices": candidates,
        "average_ess": average_ess,
        "threshold": threshold,
        "low_idx": int(low_idx),
        "low_ess_value": low_ess_value,
        "low_fraction_of_average": (low_ess_value / average_ess) if average_ess > 0.0 else math.nan,
        "all_above_threshold": bool(candidates) and all(float(ess[idx]) > threshold for idx in candidates),
    }


def local_spline_curvature_probe(
    grid: np.ndarray,
    free_energy: np.ndarray,
    center_idx: int,
    half_width: int,
) -> dict | None:
    start = max(0, int(center_idx) - int(half_width))
    stop = min(len(grid), int(center_idx) + int(half_width) + 1)
    fit_idx = [idx for idx in range(start, stop) if np.isfinite(free_energy[idx])]
    if len(fit_idx) < 4:
        return None

    x_fit = np.asarray([float(grid[idx]) for idx in fit_idx], dtype=float)
    y_fit = np.asarray([float(free_energy[idx]) for idx in fit_idx], dtype=float)
    if np.unique(x_fit).size < 4:
        return None

    try:
        spline = CubicSpline(x_fit, y_fit, bc_type="natural")
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None

    scan_x = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 401)
    curvature = np.asarray(spline(scan_x, 2), dtype=float)
    min_idx = int(np.argmin(curvature))
    x_star = float(scan_x[min_idx])
    return {
        "probe_x": float(grid[int(center_idx)]),
        "fit_xmin": float(np.min(x_fit)),
        "fit_xmax": float(np.max(x_fit)),
        "negative_curvature_x": float(x_star),
        "negative_curvature": float(curvature[min_idx]),
    }


def poly_4term_geometry(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    probe_x: float,
) -> dict | None:
    if x_fit.size < 4 or np.unique(x_fit).size < 4:
        return None
    try:
        local_x = np.asarray(x_fit, dtype=float) - float(probe_x)
        coeffs = np.polyfit(local_x, np.asarray(y_fit, dtype=float), deg=3)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    slope = float(coeffs[2])
    curvature = float(2.0 * coeffs[1])
    return {
        "slope": slope if math.isfinite(slope) else 0.0,
        "curvature": curvature if math.isfinite(curvature) else 0.0,
        "coeffs": np.asarray(coeffs, dtype=float),
        "fitted_values": np.asarray(np.polyval(coeffs, local_x), dtype=float),
    }


def poly_3term_geometry(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    probe_x: float,
) -> dict | None:
    if x_fit.size < 3 or np.unique(x_fit).size < 3:
        return None
    try:
        local_x = np.asarray(x_fit, dtype=float) - float(probe_x)
        coeffs = np.polyfit(local_x, np.asarray(y_fit, dtype=float), deg=2)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    slope = float(coeffs[1])
    curvature = float(2.0 * coeffs[0])
    return {
        "slope": slope if math.isfinite(slope) else 0.0,
        "curvature": curvature if math.isfinite(curvature) else 0.0,
        "coeffs": np.asarray(coeffs, dtype=float),
        "fitted_values": np.asarray(np.polyval(coeffs, local_x), dtype=float),
    }


def local_poly_4term_rescue_probe(
    grid: np.ndarray,
    free_energy: np.ndarray,
    center_idx: int,
    half_width: int,
    flank_points: int = 5,
) -> dict | None:
    center_idx = int(center_idx)
    if center_idx < 0 or center_idx >= len(grid):
        return None
    probe_x = float(grid[center_idx])
    fit_idx: list[int] = []
    fit_mode = ""

    if np.isfinite(free_energy[center_idx]):
        start = max(0, center_idx - int(half_width))
        stop = min(len(grid), center_idx + int(half_width) + 1)
        fit_idx = [idx for idx in range(start, stop) if np.isfinite(free_energy[idx])]
        fit_mode = "resolved_pm_around_target"
        if len(fit_idx) < 4 or np.unique(np.asarray([float(grid[idx]) for idx in fit_idx], dtype=float)).size < 4:
            fit_idx = []
            fit_mode = ""

    if not fit_idx:
        left_idx: list[int] = []
        right_idx: list[int] = []
        for idx in range(center_idx - 1, -1, -1):
            if np.isfinite(free_energy[idx]):
                left_idx.append(idx)
                if len(left_idx) >= int(flank_points):
                    break
        for idx in range(center_idx + 1, len(grid)):
            if np.isfinite(free_energy[idx]):
                right_idx.append(idx)
                if len(right_idx) >= int(flank_points):
                    break
        fit_idx = sorted(list(reversed(left_idx)) + right_idx)
        fit_mode = "bridge_from_first_resolved_left_right_poly3"

    if len(fit_idx) < 4:
        if fit_mode == "bridge_from_first_resolved_left_right_poly3" and len(fit_idx) >= 3:
            x_fit = np.asarray([float(grid[idx]) for idx in fit_idx], dtype=float)
            y_fit = np.asarray([float(free_energy[idx]) for idx in fit_idx], dtype=float)
            geometry = poly_3term_geometry(x_fit, y_fit, probe_x)
            if geometry is None:
                return None
            return {
                "probe_x": probe_x,
                "fit_mode": fit_mode,
                "fit_point_count": int(len(fit_idx)),
                "fit_xmin": float(np.min(x_fit)),
                "fit_xmax": float(np.max(x_fit)),
                "slope": float(geometry["slope"]),
                "curvature": float(geometry["curvature"]),
            }
        return None

    x_fit = np.asarray([float(grid[idx]) for idx in fit_idx], dtype=float)
    y_fit = np.asarray([float(free_energy[idx]) for idx in fit_idx], dtype=float)
    geometry = (
        poly_3term_geometry(x_fit, y_fit, probe_x)
        if fit_mode == "bridge_from_first_resolved_left_right_poly3"
        else poly_4term_geometry(x_fit, y_fit, probe_x)
    )
    if geometry is None:
        return None
    return {
        "probe_x": probe_x,
        "fit_mode": fit_mode,
        "fit_point_count": int(len(fit_idx)),
        "fit_xmin": float(np.min(x_fit)),
        "fit_xmax": float(np.max(x_fit)),
        "slope": float(geometry["slope"]),
        "curvature": float(geometry["curvature"]),
    }


def latest_window_for_prototype(
    windows: list[dict],
    prototype_window_id: int,
) -> dict | None:
    latest_window: dict | None = None
    latest_end = -1
    latest_id = -1
    for window in windows:
        window_id = int(window["window_id"])
        parent_window_id = window.get("parent_window_id", "")
        belongs_to_prototype = window_id == int(prototype_window_id)
        if parent_window_id not in ("", None):
            belongs_to_prototype = belongs_to_prototype or int(parent_window_id) == int(prototype_window_id)
        if not belongs_to_prototype:
            continue
        cumulative_end = int(window["cumulative_end"])
        if cumulative_end > latest_end or (cumulative_end == latest_end and window_id > latest_id):
            latest_window = window
            latest_end = cumulative_end
            latest_id = window_id
    return latest_window


def sample_fraction_at_grid_index(
    values: list[float],
    grid: np.ndarray,
    target_idx: int,
) -> float:
    if not values or target_idx < 0 or target_idx >= len(grid):
        return 0.0
    counts = histogram_counts(values, grid)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    return float(counts[int(target_idx)] / total)


def choose_extension_window(
    allocated_umbrellas: list[dict[str, int | float | str]],
    windows: list[dict],
    window_samples: dict[int, list[float]],
    grid: np.ndarray,
    target_idx: int,
    min_fraction: float,
) -> dict | None:
    target_x = float(grid[int(target_idx)])
    best_candidate: dict | None = None
    for prototype in allocated_umbrellas:
        prototype_id = int(prototype["window_id"])
        latest_window = latest_window_for_prototype(windows, prototype_id)
        if latest_window is None:
            continue
        latest_window_id = int(latest_window["window_id"])
        latest_samples = window_samples.get(latest_window_id, [])
        sampled_fraction = sample_fraction_at_grid_index(latest_samples, grid, target_idx)
        if sampled_fraction <= float(min_fraction):
            continue
        candidate = {
            "prototype_window_id": prototype_id,
            "prototype_side": str(prototype["side"]),
            "center_x": float(prototype["center_x"]),
            "k": float(prototype["k"]),
            "latest_window_id": latest_window_id,
            "latest_window_side": str(latest_window["side"]),
            "latest_cumulative_end": int(latest_window["cumulative_end"]),
            "sample_fraction": sampled_fraction,
            "target_x": target_x,
        }
        if best_candidate is None:
            best_candidate = candidate
            continue
        current_key = (
            sampled_fraction,
            int(latest_window["cumulative_end"]),
            -abs(float(prototype["center_x"]) - target_x),
        )
        best_key = (
            float(best_candidate["sample_fraction"]),
            int(best_candidate["latest_cumulative_end"]),
            -abs(float(best_candidate["center_x"]) - target_x),
        )
        if current_key > best_key:
            best_candidate = candidate
    return best_candidate


def choose_rescue_parent_window(
    allocated_umbrellas: list[dict[str, int | float | str]],
    windows: list[dict],
    target_x: float,
    slope: float,
) -> dict | None:
    best_window: dict | None = None
    best_key: tuple[float, float, float, int] | None = None
    target_x = float(target_x)
    slope = float(slope)
    for prototype in allocated_umbrellas:
        latest_window = latest_window_for_prototype(windows, int(prototype["window_id"]))
        if latest_window is None:
            continue
        center_x = float(latest_window["center_x"])
        preferred_side_penalty = 0.0
        if slope > 0.0:
            preferred_side_penalty = 0.0 if center_x <= target_x + 1.0e-9 else 1.0
        elif slope < 0.0:
            preferred_side_penalty = 0.0 if center_x >= target_x - 1.0e-9 else 1.0
        key = (
            preferred_side_penalty,
            abs(center_x - target_x),
            -float(latest_window["cumulative_end"]),
            int(latest_window["window_id"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_window = latest_window
    return best_window


def proposed_child_center(
    current_mean: float,
    sigma: float,
    alpha: float,
    side: str,
    grid: np.ndarray,
) -> float | None:
    if len(grid) == 0 or sigma <= 0.0:
        return None
    if side == "left":
        proposed = float(current_mean) + 2.0 * float(alpha) * float(sigma)
    else:
        proposed = float(current_mean) - 2.0 * float(alpha) * float(sigma)
    proposed = min(max(proposed, float(grid[0])), float(grid[-1]))
    idx = grid_index(proposed, grid)
    if idx is None:
        idx = int(np.clip(np.searchsorted(grid, proposed), 0, len(grid) - 1))
        if idx > 0 and abs(float(grid[idx - 1]) - proposed) <= abs(float(grid[idx]) - proposed):
            idx -= 1
    return float(grid[idx])


def estimate_local_slope(
    values: list[float],
    grid: np.ndarray,
    beta: float,
    probe_x: float,
) -> float:
    # Estimate the local PMF slope near the probe point from the parent PMF by
    # fitting a quadratic to the three resolved grid points closest to the
    # target mean and differentiating that local fit.
    counts, pmf = local_pmf_from_samples(values, grid, beta)
    resolved = np.flatnonzero((counts > 0.0) & np.isfinite(pmf))
    if resolved.size < 3:
        return 0.0
    nearest = sorted(
        (int(idx) for idx in resolved),
        key=lambda idx: abs(float(grid[idx]) - float(probe_x)),
    )[:3]
    nearest = sorted(nearest)
    x_fit = np.asarray([float(grid[idx]) for idx in nearest], dtype=float)
    y_fit = np.asarray([float(pmf[idx]) for idx in nearest], dtype=float)
    if len(np.unique(x_fit)) < 3:
        return 0.0
    coeffs = np.polyfit(x_fit, y_fit, 2)
    slope = 2.0 * float(coeffs[0]) * float(probe_x) + float(coeffs[1])
    if not math.isfinite(slope):
        return 0.0
    return abs(slope)


def choose_alpha_based_child_k(
    values: list[float],
    grid: np.ndarray,
    beta: float,
    current_mean: float,
    sigma: float,
    alpha: float,
    side: str,
) -> tuple[float, float]:
    if len(grid) > 1:
        grid_dx = abs(float(grid[1] - float(grid[0])))
    else:
        grid_dx = 1.0
    direction = 1.0 if side == "left" else -1.0
    probe_x = float(current_mean) + direction * float(alpha) * float(sigma)
    probe_x = min(max(probe_x, float(grid[0])), float(grid[-1]))
    slope = estimate_local_slope(values, grid, beta, probe_x)
    denom = max(float(alpha) * float(sigma), 0.5 * grid_dx, 1.0e-8)
    k_child = max(slope / denom, 1.0e-6)
    return k_child, probe_x


def weighted_reach(
    traj_files: list[Path],
    grid: np.ndarray,
    beta: float,
    side: str,
    current_center: float,
    ess_min: float,
) -> tuple[float | None, dict[str, list[float]]]:
    # MINES uses work-weighted accumulators so it can keep compact reduced edge
    # summaries instead of all switching trajectories.
    h = np.zeros(len(grid), dtype=float)
    q = np.zeros(len(grid), dtype=float)
    n = np.zeros(len(grid), dtype=float)
    final_works: list[float] = []

    for path in traj_files:
        rows = read_csv_rows(path)
        if not rows:
            continue
        final_works.append(float(rows[-1]["work"]))
        for row in rows:
            x = float(row["x"])
            idx = grid_index(x, grid)
            if idx is None:
                continue
            weight = math.exp(-beta * float(row["work"]))
            h[idx] += weight
            q[idx] += weight * weight
            n[idx] += 1.0

    ess = np.zeros(len(grid), dtype=float)
    valid = q > 0.0
    ess[valid] = (h[valid] * h[valid]) / q[valid]
    if side == "left":
        candidates = [float(x) for x, e in zip(grid, ess) if e >= ess_min and x > current_center + 1.0e-9]
        reach = max(candidates) if candidates else None
    else:
        candidates = [float(x) for x, e in zip(grid, ess) if e >= ess_min and x < current_center - 1.0e-9]
        reach = min(candidates) if candidates else None

    return reach, {
        "H": h.tolist(),
        "Q": q.tolist(),
        "N": n.tolist(),
        "ESS": ess.tolist(),
        "final_works": final_works,
    }


def work_overlap(forward_works: list[float], reverse_works: list[float]) -> float:
    # Compare forward work with the sign-reversed reverse work distribution,
    # which is the overlap relevant for BAR/Crooks-style connectivity checks.
    if not forward_works or not reverse_works:
        return 0.0
    reverse_mirrored = [-value for value in reverse_works]
    combined = np.asarray(forward_works + reverse_mirrored, dtype=float)
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, 41)
    f_hist, _ = np.histogram(forward_works, bins=bins, density=True)
    r_hist, _ = np.histogram(reverse_mirrored, bins=bins, density=True)
    if np.sum(f_hist) <= 0.0 or np.sum(r_hist) <= 0.0:
        return 0.0
    f_hist = f_hist / np.sum(f_hist)
    r_hist = r_hist / np.sum(r_hist)
    return float(np.sum(np.minimum(f_hist, r_hist)))


def build_common_args(ctx: dict) -> list[str]:
    pot = ctx["potential"]
    return [
        "-pot", ctx["potential_name"],
        "-one-dimension", ctx["one_dimension"],
        "-thermal_kT", str(ctx["thermal_kT"]),
        "-dt", str(ctx["dt"]),
        "-gamma", str(ctx["gamma"]),
        "-k0", str(pot["k0"]),
        "-x0", str(pot["x0"]),
        "-k1", str(pot["k1"]),
        "-x1", str(pot["x1"]),
        "-E1", str(pot["E1"]),
    ]


def run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run_eq_window(
    bin_path: str,
    ctx: dict,
    center_x: float,
    k: float,
    steps: int,
    nout: int,
    seed: int,
    out_dir: Path,
    start_xy: tuple[float, float] | None = None,
) -> tuple[list[float], str]:
    # Every adaptive window or milestone starts from the same EQ driver so the
    # raw outputs stay consistent with the fixed-window US workflow.
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_name = "eq_window.csv"
    cmd = [
        bin_path,
        *build_common_args(ctx),
        "-k", str(k),
        "-center_xy", f"{center_x},0.0",
        "-eq_out", eq_name,
        "-T_eq", str(steps),
        "-eq_nout", str(nout),
        "-eq_seed", str(seed),
        "-out_dir", str(out_dir),
        "-log", str(out_dir / "run.log"),
    ]
    if start_xy is not None:
        cmd.extend(["-eq_start_xy", f"{float(start_xy[0])},{float(start_xy[1])}"])
    run_checked(cmd)
    eq_path = out_dir / eq_name
    rows = read_csv_rows(eq_path)
    return [float(row["x"]) for row in rows], eq_name


def plan_aus_child(
    frontier: dict,
    side: str,
    grid: np.ndarray,
    grid_dx: float,
    q_next_level: float,
    alpha: float,
    fit_method: str,
    k_min: float,
    k_max: float,
    ctx: dict,
) -> tuple[dict | None, str | None]:
    """Plan one bidirectional aUS child window from the current frontier."""
    parent_values = np.asarray(frontier["samples"], dtype=float)
    if parent_values.size == 0:
        return None, f"empty_{side}_parent_window"

    parent_mean_x = float(np.mean(parent_values))
    parent_sigma_x = float(np.std(parent_values))
    parent_median_x = float(np.quantile(parent_values, 0.5))
    target_quantile = q_next_level if side == "left" else (1.0 - q_next_level)
    q_next_x = float(np.quantile(parent_values, target_quantile))
    slope = estimate_parent_window_slope(
        parent_values,
        float(frontier["center"]),
        float(frontier["k"]),
        grid,
        ctx,
        q_next_x,
        fit_method,
    )

    if side == "left":
        proposed_center = parent_median_x + alpha * (q_next_x - parent_median_x)
        proposed_center = max(proposed_center, q_next_x + grid_dx)
        derived_denom = proposed_center - q_next_x
        valid_derived = slope > 0.0 and derived_denom > 1.0e-8
        derived_k = slope / derived_denom if valid_derived else float("nan")
        fallback_flag = "k_min_fallback_for_nonpositive_slope"
    else:
        proposed_center = parent_median_x - alpha * (parent_median_x - q_next_x)
        proposed_center = min(proposed_center, q_next_x - grid_dx)
        derived_denom = q_next_x - proposed_center
        valid_derived = slope < 0.0 and derived_denom > 1.0e-8
        derived_k = (-slope) / derived_denom if valid_derived else float("nan")
        fallback_flag = "k_min_fallback_for_nonnegative_slope"

    clamp_flag = "none"
    if valid_derived:
        child_k = float(derived_k)
        child_center = float(proposed_center)
        if derived_k > k_max:
            child_k = float(k_max)
            child_center = float(q_next_x + (slope / child_k))
            clamp_flag = "k_max"
        elif derived_k < k_min:
            child_k = float(k_min)
            child_center = float(q_next_x + (slope / child_k))
            clamp_flag = "k_min"
    else:
        child_k = float(k_min)
        child_center = float(q_next_x)
        clamp_flag = fallback_flag

    if side == "left":
        child_center = max(child_center, float(frontier["center"]) + grid_dx)
        child_center = min(child_center, float(grid[-1]))
        child_center = nearest_grid_value(child_center, grid)
        if child_center <= float(frontier["center"]) + 1.0e-9:
            return None, "no_left_progress"
    else:
        child_center = min(child_center, float(frontier["center"]) - grid_dx)
        child_center = max(child_center, float(grid[0]))
        child_center = nearest_grid_value(child_center, grid)
        if child_center >= float(frontier["center"]) - 1.0e-9:
            return None, "no_right_progress"

    return (
        {
            "side": side,
            "parent_window_id": frontier["window_id"],
            "depth": int(frontier["depth"]) + 1,
            "iteration": int(frontier["depth"]) + 1,
            "center_x": float(child_center),
            "k": float(child_k),
            "mean_parent": parent_mean_x,
            "sigma_parent": parent_sigma_x,
            "median_parent_x": parent_median_x,
            "q_next_x": q_next_x,
            "target_mean_x": q_next_x,
            "local_slope": float(slope),
            "derived_k": float(derived_k) if math.isfinite(derived_k) else float("nan"),
            "k_clamped_to": clamp_flag,
            "alpha": float(alpha),
            "fit_method": fit_method,
            "q_next": float(q_next_level),
            "k_min": float(k_min),
            "k_max": float(k_max),
        },
        None,
    )


def run_neq_edge(
    bin_path: str,
    ctx: dict,
    left_center: float,
    right_center: float,
    eq_left: Path,
    eq_right: Path,
    k: float,
    n_traj_per_direction: int,
    t_neq: int,
    nout: int,
    seed: int,
    out_dir: Path,
) -> None:
    # MINES edges reuse the standard NES switching implementation between two
    # milestone centers.
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        bin_path,
        *build_common_args(ctx),
        "-k", str(k),
        "-k_midscale", str(ctx["nes_screen"]["fixed"]["k_midscale"]),
        "-A_center", f"{left_center},0.0",
        "-B_center", f"{right_center},0.0",
        "-eq0", str(eq_left),
        "-eq1", str(eq_right),
        "-N_neq", str(n_traj_per_direction),
        "-T_neq", str(t_neq),
        "-neq_nout", str(nout),
        "-neq_seed", str(seed),
        "-out_dir", str(out_dir),
        "-log", str(out_dir / "neq.log"),
    ]
    run_checked(cmd)


def shift_finite_to_zero(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    if np.any(finite):
        result[finite] -= float(np.min(result[finite]))
    return result


def analytic_doublewell_profile_grid(grid: np.ndarray, ctx: dict) -> np.ndarray:
    pot = ctx["potential"]
    beta = 1.0 / float(ctx["thermal_kT"])
    xs = np.asarray(grid, dtype=float)
    u0 = float(pot["k0"]) * (xs - float(pot["x0"])) * (xs - float(pot["x0"]))
    u1 = float(pot["k1"]) * (xs - float(pot["x1"])) * (xs - float(pot["x1"]))
    log_t0 = -beta * u0
    log_t1 = -beta * u1 - float(pot["E1"])
    log_max = np.maximum(log_t0, log_t1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        profile = -(log_max + np.log(np.exp(log_t0 - log_max) + np.exp(log_t1 - log_max))) / beta
    return shift_finite_to_zero(profile)


def hs_reconstruct_oneway_forward(
    traj_paths: list[Path],
    grid: np.ndarray,
    ctx: dict,
    base_k: float,
    left_x: float,
    right_x: float,
) -> np.ndarray:
    beta = 1.0 / float(ctx["thermal_kT"])
    dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    xs = np.asarray(grid, dtype=float)
    lambdas: list[float] | None = None
    log_sum_w: np.ndarray | None = None
    log_sum_hist: np.ndarray | None = None
    n_traj = 0

    for path in traj_paths:
        rows = read_csv_rows(path)
        if not rows:
            continue
        if lambdas is None:
            lambdas = [float(row["lambda"]) for row in rows]
            log_sum_w = np.full(len(lambdas), -np.inf, dtype=float)
            log_sum_hist = np.full((len(lambdas), len(xs)), -np.inf, dtype=float)
        assert log_sum_w is not None
        assert log_sum_hist is not None
        n_traj += 1
        for time_idx, row in enumerate(rows):
            log_weight = -beta * float(row["work"])
            log_sum_w[time_idx] = np.logaddexp(log_sum_w[time_idx], log_weight)
            idx = grid_index(float(row["x"]), grid)
            if idx is not None:
                log_sum_hist[time_idx, idx] = np.logaddexp(log_sum_hist[time_idx, idx], log_weight)

    if lambdas is None or log_sum_w is None or log_sum_hist is None or n_traj <= 0:
        return np.full(len(xs), np.nan, dtype=float)

    log_numerator_terms = log_sum_hist - log_sum_w[:, None]
    log_denominator_terms = np.full((len(lambdas), len(xs)), -np.inf, dtype=float)
    log_n_traj = math.log(float(n_traj))
    for time_idx, lamb in enumerate(lambdas):
        if not math.isfinite(float(log_sum_w[time_idx])):
            continue
        center = float(left_x) + float(lamb) * (float(right_x) - float(left_x))
        log_denominator_terms[time_idx] = (
            log_n_traj
            - float(log_sum_w[time_idx])
            - beta * 0.5 * float(base_k) * (xs - center) * (xs - center)
        )

    log_numerator = logsumexp_np(log_numerator_terms, axis=0)
    log_denominator = logsumexp_np(log_denominator_terms, axis=0)
    log_density = log_numerator - log_denominator
    valid = np.isfinite(log_density)
    if np.any(valid):
        log_norm = float(logsumexp_np(log_density[valid])) + math.log(dx)
        log_density[valid] -= log_norm

    pmf = np.full(len(xs), np.nan, dtype=float)
    pmf[valid] = -float(ctx["thermal_kT"]) * log_density[valid]
    return shift_finite_to_zero(pmf)


def pmf_to_density(pmf: np.ndarray, dx: float, kT: float) -> np.ndarray:
    density = np.zeros(len(pmf), dtype=float)
    finite = np.isfinite(pmf)
    if np.any(finite):
        density[finite] = np.exp(-pmf[finite] / float(kT))
        norm = float(np.sum(density) * dx)
        if norm > 0.0:
            density /= norm
    return density


def log_normalized_density_from_pmf(pmf: np.ndarray, dx: float, kT: float) -> np.ndarray:
    log_density = np.full(len(pmf), -np.inf, dtype=float)
    finite = np.isfinite(pmf)
    if not np.any(finite):
        return log_density
    log_weights = -np.asarray(pmf[finite], dtype=float) / float(kT)
    log_norm = float(logsumexp_np(log_weights)) + math.log(dx)
    log_density[finite] = log_weights - log_norm
    return log_density


def average_pmfs_by_density(pmfs: list[np.ndarray], dx: float, kT: float) -> np.ndarray:
    if not pmfs:
        return np.zeros(0, dtype=float)
    log_density_sum = np.full(len(pmfs[0]), -np.inf, dtype=float)
    used = 0
    for pmf in pmfs:
        log_local = log_normalized_density_from_pmf(np.asarray(pmf, dtype=float), dx, kT)
        if not np.any(np.isfinite(log_local)):
            continue
        log_density_sum = np.logaddexp(log_density_sum, log_local)
        used += 1
    if used <= 0:
        return np.full(len(pmfs[0]), np.nan, dtype=float)
    log_density = log_density_sum - math.log(float(used))
    finite = np.isfinite(log_density)
    if np.any(finite):
        log_norm = float(logsumexp_np(log_density[finite])) + math.log(dx)
        log_density[finite] -= log_norm
    result = np.full(len(log_density), np.nan, dtype=float)
    result[finite] = -float(kT) * log_density[finite]
    return shift_finite_to_zero(result)


def write_protocol_path(path: Path, centers: list[float], ks: list[float]) -> None:
    if len(centers) != len(ks):
        raise ValueError("MiNES protocol path centers and spring lists must have the same length.")
    n = max(len(centers) - 1, 1)
    rows = []
    for idx, (center, k_value) in enumerate(zip(centers, ks)):
        rows.append(
            {
                "lambda": float(idx) / float(n),
                "x0": float(center),
                "y0": 0.0,
                "k": float(k_value),
            }
        )
    write_csv(path, ["lambda", "x0", "y0", "k"], rows)


def run_neq_protocol_single_start(
    bin_path: str,
    ctx: dict,
    start_eq_path: Path,
    fallback_eq_path: Path,
    path_csv: Path,
    seed: int,
    out_dir: Path,
) -> None:
    rows = read_csv_rows(path_csv)
    if len(rows) < 2:
        raise RuntimeError(f"MiNES reversible path needs at least two path rows: {path_csv}")
    t_neq = len(rows) - 2
    nout = max(t_neq + 1, 1)
    start_center = float(rows[0]["x0"])
    end_center = float(rows[-1]["x0"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        bin_path,
        *build_common_args(ctx),
        "-k", str(float(rows[0]["k"])),
        "-k_midscale", "1.0",
        "-A_center", f"{start_center},0.0",
        "-B_center", f"{end_center},0.0",
        "-eq0", str(start_eq_path),
        "-eq1", str(fallback_eq_path),
        "-fpath", str(path_csv),
        "-N_neq", "1",
        "-T_neq", str(t_neq),
        "-neq_nout", str(nout),
        "-neq_seed", str(seed),
        "-out_dir", str(out_dir),
        "-log", str(out_dir / "neq.log"),
    ]
    run_checked(cmd)


def read_protocol_path_rows(path: Path) -> list[dict[str, float]]:
    rows = read_csv_rows(path)
    result: list[dict[str, float]] = []
    for row in rows:
        result.append(
            {
                "lambda": float(row["lambda"]),
                "center_x": float(row["x0"]),
                "center_y": float(row.get("y0", "0.0") or 0.0),
                "k": float(row["k"]),
            }
        )
    return result


def hs_reconstruct_oneway_path_rows(
    trajectories: list[list[dict[str, str]]],
    path_rows: list[dict[str, float]],
    grid: np.ndarray,
    ctx: dict,
) -> np.ndarray:
    if not trajectories or not path_rows:
        return np.full(len(grid), np.nan, dtype=float)
    n_time = min(len(path_rows), min(len(traj) for traj in trajectories))
    beta = 1.0 / float(ctx["thermal_kT"])
    dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    xs = np.asarray(grid, dtype=float)
    log_sum_w = np.full(n_time, -np.inf, dtype=float)
    log_sum_hist = np.full((n_time, len(xs)), -np.inf, dtype=float)
    n_traj = 0
    for traj in trajectories:
        if len(traj) < n_time:
            continue
        n_traj += 1
        for time_idx in range(n_time):
            row = traj[time_idx]
            log_weight = -beta * float(row["work"])
            log_sum_w[time_idx] = np.logaddexp(log_sum_w[time_idx], log_weight)
            idx = grid_index(float(row["x"]), grid)
            if idx is not None:
                log_sum_hist[time_idx, idx] = np.logaddexp(log_sum_hist[time_idx, idx], log_weight)
    if n_traj <= 0:
        return np.full(len(xs), np.nan, dtype=float)
    log_numerator_terms = np.full((n_time, len(xs)), -np.inf, dtype=float)
    log_denominator_terms = np.full((n_time, len(xs)), -np.inf, dtype=float)
    log_n_traj = math.log(float(n_traj))
    for time_idx in range(n_time):
        if not math.isfinite(float(log_sum_w[time_idx])):
            continue
        center = float(path_rows[time_idx]["center_x"])
        k_value = float(path_rows[time_idx]["k"])
        log_numerator_terms[time_idx] = log_sum_hist[time_idx] - log_n_traj - log_sum_w[time_idx]
        log_denominator_terms[time_idx] = (
            -log_sum_w[time_idx]
            - beta * 0.5 * k_value * (xs - center) * (xs - center)
        )
    log_density = logsumexp_np(log_numerator_terms, axis=0) - logsumexp_np(log_denominator_terms, axis=0)
    valid = np.isfinite(log_density)
    if np.any(valid):
        log_norm = float(logsumexp_np(log_density[valid])) + math.log(dx)
        log_density[valid] -= log_norm
    pmf = np.full(len(xs), np.nan, dtype=float)
    pmf[valid] = -float(ctx["thermal_kT"]) * log_density[valid]
    return shift_finite_to_zero(pmf)


def align_pmf_to_analytic_anchor(
    pmf: np.ndarray,
    analytic: np.ndarray,
    grid: np.ndarray,
    anchor_x: float,
) -> tuple[np.ndarray, float]:
    aligned = np.asarray(pmf, dtype=float).copy()
    idx = grid_index(anchor_x, grid)
    if idx is None:
        return aligned, float("nan")
    if not math.isfinite(float(aligned[idx])) or not math.isfinite(float(analytic[idx])):
        return aligned, float("nan")
    shift = float(aligned[idx] - analytic[idx])
    finite = np.isfinite(aligned)
    if np.any(finite):
        aligned[finite] -= shift
    return aligned, shift


def align_pmf_to_value(
    pmf: np.ndarray,
    grid: np.ndarray,
    anchor_x: float,
    target_value: float,
) -> tuple[np.ndarray, float]:
    aligned = np.asarray(pmf, dtype=float).copy()
    idx = grid_index(anchor_x, grid)
    if idx is None:
        return aligned, float("nan")
    if not math.isfinite(float(aligned[idx])):
        return aligned, float("nan")
    shift = float(aligned[idx] - float(target_value))
    finite = np.isfinite(aligned)
    if np.any(finite):
        aligned[finite] -= shift
    return aligned, shift


def retained_tail_rows(eq_rows: list[dict[str, str]], keep_fraction: float) -> list[dict[str, str]]:
    if not eq_rows:
        return []
    discard_count = int(math.floor(len(eq_rows) * (1.0 - float(keep_fraction))))
    discard_count = min(max(discard_count, 0), max(len(eq_rows) - 1, 0))
    return list(eq_rows[discard_count:])


def x_values_from_rows(rows: list[dict[str, str]], field: str = "x") -> list[float]:
    return [float(row[field]) for row in rows]


def x_most_from_tail_rows(tail_rows: list[dict[str, str]], grid: np.ndarray) -> float:
    if not tail_rows:
        return float("nan")
    counts = histogram_counts(x_values_from_rows(tail_rows), grid)
    if counts.size <= 0:
        return float("nan")
    return float(grid[int(np.argmax(counts))])


def sample_tail_rows(tail_rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    if not tail_rows or count <= 0:
        return []
    draw_count = min(int(count), len(tail_rows))
    rng = np.random.default_rng(seed)
    draw_indices = rng.choice(len(tail_rows), size=draw_count, replace=False)
    return [tail_rows[int(idx)] for idx in draw_indices.tolist()]


def write_path_linear_sqrtk(
    path: Path,
    left_center: float,
    right_center: float,
    left_k: float,
    right_k: float,
    t_neq: int,
) -> list[dict[str, float]]:
    n_path_rows = int(t_neq) + 2
    fractions = np.linspace(0.0, 1.0, n_path_rows)
    centers = (float(left_center) + fractions * (float(right_center) - float(left_center))).tolist()
    sqrt_ks = np.sqrt(float(left_k)) + fractions * (np.sqrt(float(right_k)) - np.sqrt(float(left_k)))
    ks = (sqrt_ks * sqrt_ks).tolist()
    write_protocol_path(path, centers, ks)
    return read_protocol_path_rows(path)


def collect_saved_neq_candidates(
    trajectories: list[list[dict[str, str]]],
    side_label: str,
) -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = []
    for draw_idx, traj_rows in enumerate(trajectories):
        for row_idx, row in enumerate(traj_rows):
            candidates.append(
                {
                    "source_side": side_label,
                    "draw_idx": int(draw_idx),
                    "row_idx": int(row_idx),
                    "step": int(float(row["step"])),
                    "x": float(row["x"]),
                    "y": float(row.get("y", "0.0") or 0.0),
                    "work": float(row["work"]),
                }
            )
    return candidates


def select_low_work_start_candidate(
    candidates: list[dict[str, float | int | str]],
    target_x: float,
    grid: np.ndarray,
) -> dict[str, float | int | str]:
    if not candidates:
        raise RuntimeError("MiNES offspring initialization needs at least one saved NES candidate frame.")
    target_idx = grid_index(float(target_x), grid)
    if target_idx is None:
        target_idx = int(np.argmin(np.abs(grid - float(target_x))))
    indexed: list[tuple[int, dict[str, float | int | str]]] = []
    for candidate in candidates:
        idx = grid_index(float(candidate["x"]), grid)
        if idx is None:
            idx = int(np.argmin(np.abs(grid - float(candidate["x"]))))
        indexed.append((idx, candidate))
    exact = [candidate for idx, candidate in indexed if idx == target_idx]
    if exact:
        pool = exact
        selected_idx = target_idx
        selection_mode = "target_bin"
    else:
        min_distance = min(abs(idx - target_idx) for idx, _ in indexed)
        pool = [candidate for idx, candidate in indexed if abs(idx - target_idx) == min_distance]
        selected_idx = min(
            int(np.argmin(np.abs(grid - float(candidate["x"]))))
            for candidate in pool
        )
        selection_mode = "closest_bin"
    chosen = min(pool, key=lambda row: (float(row["work"]), abs(float(row["x"]) - float(target_x))))
    result = dict(chosen)
    result["target_x"] = float(target_x)
    result["target_bin_x"] = float(grid[int(target_idx)])
    result["selected_bin_x"] = float(grid[int(selected_idx)])
    result["selection_mode"] = selection_mode
    return result


def run_single_start_paths(
    bin_path: str,
    ctx: dict,
    path_file: Path,
    start_rows: list[dict[str, str]],
    fallback_xy: tuple[float, float],
    seed_base: int,
    out_dir: Path,
) -> tuple[list[dict], list[dict], list[list[dict[str, str]]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = out_dir / "fallback_eq.csv"
    write_csv(fallback_path, ["x", "y"], [{"x": float(fallback_xy[0]), "y": float(fallback_xy[1])}])
    draw_rows: list[dict] = []
    endpoint_rows: list[dict] = []
    trajectories: list[list[dict[str, str]]] = []
    for draw_idx, source_row in enumerate(start_rows):
        start_x = float(source_row["x"])
        start_y = float(source_row.get("y", "0.0") or 0.0)
        draw_rows.append(
            {
                "draw_idx": int(draw_idx),
                "source_tail_index": int(source_row.get("source_tail_index", draw_idx)),
                "source_step": int(float(source_row.get("step", source_row.get("source_step", 0)) or 0.0)),
                "start_x": start_x,
                "start_y": start_y,
                "base_u": float(source_row.get("base_u", "nan") or "nan"),
                "bias_u": float(source_row.get("bias_u", "nan") or "nan"),
            }
        )
        start_path = out_dir / "starts" / f"start_{draw_idx:03d}.csv"
        write_csv(start_path, ["x", "y"], [{"x": start_x, "y": start_y}])
        traj_dir = out_dir / f"traj_{draw_idx:03d}"
        run_neq_protocol_single_start(
            bin_path,
            ctx,
            start_path,
            fallback_path,
            path_file,
            seed_base + draw_idx,
            traj_dir,
        )
        traj_path = traj_dir / "neq_fwd_0.csv"
        traj_rows = read_csv_rows(traj_path)
        if not traj_rows:
            raise RuntimeError(f"MiNES protocol produced no trajectory rows for draw {draw_idx} at {out_dir}.")
        trajectories.append(traj_rows)
        final_row = traj_rows[-1]
        endpoint_rows.append(
            {
                "draw_idx": int(draw_idx),
                "start_x": start_x,
                "start_y": start_y,
                "final_step": int(float(final_row["step"])),
                "final_lambda": float(final_row["lambda"]),
                "final_x": float(final_row["x"]),
                "final_y": float(final_row.get("y", "0.0") or 0.0),
                "final_work": float(final_row["work"]),
                "traj_file": str(traj_path.relative_to(out_dir.parent)),
            }
        )
    return draw_rows, endpoint_rows, trajectories


def estimate_hs_child_design_slope(
    trajectories: list[list[dict[str, str]]],
    path_rows: list[dict[str, float]],
    final_x: np.ndarray,
    target_x: float,
    q_next_level: float,
    grid: np.ndarray,
    ctx: dict,
) -> tuple[float, str]:
    if not trajectories or not path_rows or final_x.size <= 0:
        return float("nan"), "hs_unavailable"
    hs_pmf = hs_reconstruct_oneway_path_rows(trajectories, path_rows, grid, ctx)
    if not np.any(np.isfinite(hs_pmf)):
        return float("nan"), "hs_all_nan"
    lower_q = float(np.quantile(final_x, 1.0 - float(q_next_level)))
    upper_q = float(np.quantile(final_x, float(q_next_level)))
    interval_lo = min(lower_q, upper_q)
    interval_hi = max(lower_q, upper_q)
    fit_mask = np.isfinite(hs_pmf) & (grid >= interval_lo) & (grid <= interval_hi)
    if int(np.count_nonzero(fit_mask)) < 4:
        return float("nan"), "hs_interval_underresolved"
    x_fit = np.asarray(grid[fit_mask], dtype=float)
    y_fit = np.asarray(hs_pmf[fit_mask], dtype=float)
    slope, _ = evaluate_local_fit(x_fit, y_fit, float(target_x), "poly_4term_parent")
    if not math.isfinite(float(slope)):
        return float("nan"), "hs_slope_nan"
    return float(slope), "ok"


def design_force_matched_child(
    side: str,
    final_x: np.ndarray,
    parent_tail_rows: list[dict[str, str]],
    trajectories: list[list[dict[str, str]]],
    path_rows: list[dict[str, float]],
    parent_center: float,
    parent_x_most: float,
    opposite_center: float,
    opposite_k: float,
    alpha: float,
    x_method_leap: str,
    x_leap: float,
    q_next_level: float,
    k_min: float,
    k_max: float,
    target_rule: str,
    k_method_leap: str,
    grid: np.ndarray,
    ctx: dict,
 ) -> dict[str, float | str | bool]:
    if final_x.size <= 0:
        raise RuntimeError("MiNES child design needs at least one end-of-protocol NES sample.")
    parent_tail_x = np.asarray([float(row["x"]) for row in parent_tail_rows], dtype=float)
    if parent_tail_x.size <= 0:
        raise RuntimeError("MiNES child design needs at least one retained parent EQ-tail sample.")
    q50 = float(np.quantile(final_x, 0.5))
    anchor_level = float(q_next_level if side == "left" else (1.0 - q_next_level))
    q_anchor = float(np.quantile(final_x, anchor_level))
    parent_q50 = float(np.quantile(parent_tail_x, 0.5))
    parent_q_anchor = float(np.quantile(parent_tail_x, anchor_level))
    if target_rule == "target-next":
        target_x = q_anchor
        barrier_target_x = parent_q_anchor
    elif target_rule == "target-median":
        target_x = q50
        barrier_target_x = parent_q50
    else:
        raise ValueError(f"Unsupported MiNES target rule: {target_rule}")

    local_slope_dx = float(parent_x_most - parent_center)
    if local_slope_dx > 1.0e-8:
        local_slope_sign = "negative"
    elif local_slope_dx < -1.0e-8:
        local_slope_sign = "positive"
    else:
        local_slope_sign = "flat"

    progress_favorable = (
        float(parent_x_most) > float(parent_center) + 1.0e-8
        if side == "left"
        else float(parent_x_most) < float(parent_center) - 1.0e-8
    )
    target_source = "forward_endpoint_distribution"
    design_mode = "endpoint-driven"
    design_reason = "eq_local_slope_not_favorable_for_progress"
    if progress_favorable:
        design_mode = "barrier-crossing"
        design_reason = "eq_local_slope_favorable_for_progress"
        target_source = "parent_eq_tail"
        target_x = float(barrier_target_x)
        q50 = float(parent_q50)
        q_anchor = float(parent_q_anchor)
        if x_method_leap == "leap-alpha":
            if side == "left":
                center_raw = float(target_x + float(alpha) * (q_anchor - q50))
                denom = float(center_raw - target_x)
                matched_force = float(opposite_k * (opposite_center - target_x))
            else:
                center_raw = float(target_x - float(alpha) * (q50 - q_anchor))
                denom = float(target_x - center_raw)
                matched_force = float(opposite_k * (target_x - opposite_center))
        elif x_method_leap == "leap-fixed":
            if side == "left":
                center_raw = float(target_x + float(x_leap))
                denom = float(center_raw - target_x)
                matched_force = float(opposite_k * (opposite_center - target_x))
            else:
                center_raw = float(target_x - float(x_leap))
                denom = float(target_x - center_raw)
                matched_force = float(opposite_k * (target_x - opposite_center))
        else:
            raise ValueError(f"Unsupported MiNES leap rule: {x_method_leap}")
        return {
            "target_x": float(target_x),
            "anchor_x": float(q_anchor),
            "q50_x": float(q50),
            "anchor_level": float(anchor_level),
            "target_source": str(target_source),
            "barrier_crossing": True,
            "progress_favorable": bool(progress_favorable),
            "x_method_leap": str(x_method_leap),
            "x_leap": float(x_leap),
            "center_raw": float(center_raw),
            "center_x": float(center_raw),
            "matched_force": float(matched_force),
            "gap": float(denom),
            "raw_k": float("nan"),
            "k": float(k_min),
            "k_clamped_to": "barrier_crossing_k_min",
            "design_mode": design_mode,
            "design_reason": design_reason,
            "local_slope_sign": local_slope_sign,
            "local_slope_dx": float(local_slope_dx),
            "k_method_requested": str(k_method_leap),
            "k_method_applied": "barrier_crossing_k_min",
            "slope_at_target": float("nan"),
            "slope_status": "barrier_crossing_mode",
            "k_fallback_reason": "",
        }

    if x_method_leap == "leap-alpha":
        if side == "left":
            center_raw = float(target_x + float(alpha) * (q_anchor - q50))
        else:
            center_raw = float(target_x - float(alpha) * (q50 - q_anchor))
    elif x_method_leap == "leap-fixed":
        if side == "left":
            center_raw = float(target_x + float(x_leap))
        else:
            center_raw = float(target_x - float(x_leap))
    else:
        raise ValueError(f"Unsupported MiNES leap rule: {x_method_leap}")

    force_matched_force = (
        float(opposite_k * (opposite_center - target_x))
        if side == "left"
        else float(opposite_k * (target_x - opposite_center))
    )
    force_denom = (
        float(center_raw - target_x)
        if side == "left"
        else float(target_x - center_raw)
    )

    applied_k_method = str(k_method_leap)
    k_fallback_reason = ""
    slope_at_target = float("nan")
    slope_status = "not_requested"

    if str(k_method_leap) == "Slope-matching":
        slope_at_target, slope_status = estimate_hs_child_design_slope(
            trajectories,
            path_rows,
            final_x,
            float(target_x),
            float(q_next_level),
            grid,
            ctx,
        )
        if not math.isfinite(float(slope_at_target)):
            applied_k_method = "Force-matching"
            k_fallback_reason = str(slope_status)
        elif side == "left" and slope_at_target <= 0.0:
            applied_k_method = "Force-matching"
            k_fallback_reason = "slope_wrong_sign_left"
        elif side == "right" and slope_at_target >= 0.0:
            applied_k_method = "Force-matching"
            k_fallback_reason = "slope_wrong_sign_right"
        else:
            slope_status = "ok"
    elif str(k_method_leap) != "Force-matching":
        raise ValueError(f"Unsupported MiNES stiffness rule: {k_method_leap}")

    raw_k = float("nan")
    center_final = float(center_raw)
    k_final = float(k_min)
    clamp_flag = "invalid_keep_leap_center"
    matched_force = force_matched_force
    denom = force_denom
    if applied_k_method == "Slope-matching":
        matched_force = float("nan")
        denom = float(center_raw - target_x)
        if abs(denom) > 1.0e-12:
            raw_k = float(slope_at_target / denom)
        if math.isfinite(raw_k) and raw_k > 0.0:
            k_final = float(min(max(raw_k, k_min), k_max))
            clamp_flag = "none"
            if raw_k < k_min:
                clamp_flag = "k_min"
            elif raw_k > k_max:
                clamp_flag = "k_max"
            if clamp_flag != "none":
                center_final = float(target_x + slope_at_target / k_final)
        else:
            applied_k_method = "Force-matching"
            k_fallback_reason = str(k_fallback_reason or "invalid_slope_k")

    if applied_k_method == "Force-matching":
        matched_force = force_matched_force
        denom = force_denom
        raw_k = float("nan")
        center_final = float(center_raw)
        k_final = float(k_min)
        clamp_flag = "force_match_invalid_keep_leap_center"
        if matched_force > 0.0 and denom > 1.0e-12:
            raw_k = float(matched_force / denom)
            if math.isfinite(raw_k):
                k_final = float(min(max(raw_k, k_min), k_max))
                clamp_flag = "none"
                if raw_k < k_min:
                    clamp_flag = "k_min"
                elif raw_k > k_max:
                    clamp_flag = "k_max"
                if side == "left" and clamp_flag != "none":
                    center_final = float(target_x + (opposite_k / k_final) * (opposite_center - target_x))
                elif side == "right" and clamp_flag != "none":
                    center_final = float(target_x - (opposite_k / k_final) * (target_x - opposite_center))
                else:
                    center_final = float(center_raw)
        else:
            center_final = float(center_raw)

    return {
        "target_x": float(target_x),
        "anchor_x": float(q_anchor),
        "q50_x": float(q50),
        "anchor_level": float(anchor_level),
        "target_source": str(target_source),
        "barrier_crossing": False,
        "progress_favorable": bool(progress_favorable),
        "x_method_leap": str(x_method_leap),
        "x_leap": float(x_leap),
        "center_raw": float(center_raw),
        "center_x": float(center_final),
        "matched_force": float(matched_force),
        "gap": float(denom),
        "raw_k": float(raw_k),
        "k": float(k_final),
        "k_clamped_to": clamp_flag,
        "design_mode": design_mode,
        "design_reason": design_reason,
        "local_slope_sign": local_slope_sign,
        "local_slope_dx": float(local_slope_dx),
        "k_method_requested": str(k_method_leap),
        "k_method_applied": str(applied_k_method),
        "slope_at_target": float(slope_at_target),
        "slope_status": str(slope_status),
        "k_fallback_reason": str(k_fallback_reason),
    }


def append_terminal_bias_rows(
    trajectories: list[list[dict[str, str]]],
    protocol_end_center: float,
    protocol_end_k: float,
    child_center: float,
    child_k: float,
) -> tuple[list[list[dict[str, str]]], list[dict]]:
    synthetic_trajectories: list[list[dict[str, str]]] = []
    endpoint_rows: list[dict] = []
    for draw_idx, traj_rows in enumerate(trajectories):
        final_row = traj_rows[-1]
        x_t = float(final_row["x"])
        y_t = float(final_row.get("y", "0.0") or 0.0)
        terminal_old = 0.5 * float(protocol_end_k) * (x_t - float(protocol_end_center)) ** 2
        terminal_new = 0.5 * float(child_k) * (x_t - float(child_center)) ** 2
        appended_work = float(final_row["work"]) + (terminal_new - terminal_old)
        synthetic_rows = list(traj_rows)
        synthetic_rows.append(
            {
                "step": str(int(float(final_row["step"])) + 1),
                "lambda": "1.0",
                "x": f"{x_t:.10f}",
                "y": f"{y_t:.10f}",
                "base_u": final_row.get("base_u", ""),
                "bias_u": f"{terminal_new:.10f}",
                "work": f"{appended_work:.10f}",
            }
        )
        synthetic_trajectories.append(synthetic_rows)
        endpoint_rows.append(
            {
                "draw_idx": int(draw_idx),
                "final_step": int(float(synthetic_rows[-1]["step"])),
                "final_lambda": 1.0,
                "final_x": x_t,
                "final_y": y_t,
                "appended_work": appended_work,
                "terminal_bias_old": terminal_old,
                "terminal_bias_new": terminal_new,
            }
        )
    return synthetic_trajectories, endpoint_rows


def build_mines_child_from_parent(
    system_root: Path,
    ctx: dict,
    seed: int,
    bin_path: str,
    out_root: Path,
    side: str,
    label: str,
    parent_name: str,
    child_name: str,
    parent_center: float,
    parent_k: float,
    parent_eq_rows: list[dict[str, str]],
    parent_tail_rows: list[dict[str, str]],
    parent_eq_ref: str,
    parent_tail_ref: str,
    protocol_end_center: float,
    protocol_k: float,
    k_min: float,
    k_max: float,
    t_nes: int,
    use_tail: float,
    n_nes: int,
    alpha: float,
    q_next_level: float,
    target_rule: str,
    seed_offset: int,
) -> dict:
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported MiNES child-build side: {side}")
    if not parent_eq_rows or not parent_tail_rows:
        raise RuntimeError(f"Missing parent EQ rows for {parent_name} -> {child_name}.")

    out_root.mkdir(parents=True, exist_ok=True)
    window0_dir = out_root / "window_0"
    write_csv(window0_dir / "eq_window.csv", list(parent_eq_rows[0].keys()), parent_eq_rows)
    write_csv(window0_dir / "eq_tail.csv", list(parent_tail_rows[0].keys()), parent_tail_rows)

    grid = build_grid(
        float(ctx["grid"]["xmin"]),
        float(ctx["grid"]["xmax"]),
        float(ctx["grid"]["dx"]),
    )
    grid_dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0

    rng = np.random.default_rng(seed + seed_offset)
    draw_count = min(int(n_nes), len(parent_tail_rows))
    draw_indices = rng.choice(len(parent_tail_rows), size=draw_count, replace=False)

    nes_dir = out_root / "nes"
    nes_dir.mkdir(parents=True, exist_ok=True)
    dummy_eq_path = nes_dir / "eq_protocol_end_dummy.csv"
    write_csv(dummy_eq_path, ["x", "y"], [{"x": float(protocol_end_center), "y": 0.0}])

    draw_rows: list[dict] = []
    endpoint_rows: list[dict] = []
    traj_paths: list[Path] = []
    for draw_idx, source_tail_index in enumerate(draw_indices.tolist()):
        source_row = parent_tail_rows[int(source_tail_index)]
        start_x = float(source_row["x"])
        start_y = float(source_row.get("y", "0.0") or 0.0)
        draw_rows.append(
            {
                "draw_idx": int(draw_idx),
                "source_tail_index": int(source_tail_index),
                "source_step": int(float(source_row.get("step", "0") or 0.0)),
                "x": start_x,
                "y": start_y,
                "base_u": float(source_row.get("base_u", "nan") or "nan"),
                "bias_u": float(source_row.get("bias_u", "nan") or "nan"),
            }
        )

        start_path = nes_dir / "starts" / f"start_{draw_idx:03d}.csv"
        write_csv(start_path, ["x", "y"], [{"x": start_x, "y": start_y}])

        traj_dir = nes_dir / f"traj_{draw_idx:03d}"
        run_neq_edge(
            bin_path,
            ctx,
            parent_center,
            protocol_end_center,
            start_path,
            dummy_eq_path,
            protocol_k,
            1,
            t_nes,
            t_nes,
            seed + seed_offset + 1000 + draw_idx,
            traj_dir,
        )
        traj_path = traj_dir / "neq_fwd_0.csv"
        traj_rows = read_csv_rows(traj_path)
        if not traj_rows:
            raise RuntimeError(
                f"MiNES child build produced no forward trajectory rows for {parent_name} -> {child_name}, draw {draw_idx}."
            )
        traj_paths.append(traj_path)
        final_row = traj_rows[-1]
        endpoint_rows.append(
            {
                "draw_idx": int(draw_idx),
                "start_x": start_x,
                "start_y": start_y,
                "final_step": int(float(final_row["step"])),
                "final_lambda": float(final_row["lambda"]),
                "final_x": float(final_row["x"]),
                "final_y": float(final_row.get("y", "0.0") or 0.0),
                "final_work": float(final_row["work"]),
                "traj_file": str(traj_path.relative_to(out_root)),
            }
        )

    write_csv(
        nes_dir / "drawn_start_samples.csv",
        ["draw_idx", "source_tail_index", "source_step", "x", "y", "base_u", "bias_u"],
        draw_rows,
    )
    write_csv(
        nes_dir / "forward_endpoints.csv",
        ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
        endpoint_rows,
    )

    hs_pmf = hs_reconstruct_oneway_forward(
        traj_paths,
        grid,
        ctx,
        protocol_k,
        parent_center,
        protocol_end_center,
    )
    analytic_pmf = analytic_doublewell_profile_grid(grid, ctx)
    write_csv(
        out_root / "oneway_hs_pmf.csv",
        ["x", "hs_pmf", "analytic_pmf"],
        [
            {
                "x": float(x),
                "hs_pmf": "" if not math.isfinite(float(hs)) else float(hs),
                "analytic_pmf": float(analytic),
            }
            for x, hs, analytic in zip(grid.tolist(), hs_pmf.tolist(), analytic_pmf.tolist())
        ],
    )

    final_x = np.asarray([float(row["final_x"]) for row in endpoint_rows], dtype=float)
    if final_x.size == 0:
        raise RuntimeError(f"MiNES child build produced no final endpoints for {parent_name} -> {child_name}.")
    x_q50 = float(np.quantile(final_x, 0.5))
    anchor_quantile_level = q_next_level if side == "left" else (1.0 - q_next_level)
    x_anchor = float(np.quantile(final_x, anchor_quantile_level))
    if target_rule == "median_q0.5":
        x_target = x_q50
    elif target_rule == "q_next_anchor":
        x_target = x_anchor
    else:
        raise ValueError(f"Unsupported MiNES child target rule: {target_rule}")
    if side == "left":
        x_next_raw = float(x_target + alpha * (x_anchor - x_target))
        if target_rule == "q_next_anchor":
            x_next_raw = float(x_q50 + alpha * (x_anchor - x_q50))
        x_next = min(max(x_next_raw, float(grid[0])), float(grid[-1]))
        if x_next <= x_target + 1.0e-8:
            x_next = min(float(grid[-1]), float(x_target + grid_dx))
        if x_next <= parent_center + 1.0e-8:
            x_next = min(float(grid[-1]), float(parent_center + grid_dx))
        matched_force = float(protocol_k * (protocol_end_center - x_target))
        gap = float(x_next - x_target)
        stiffness_formula = "k_protocol_end_times_x_protocol_end_minus_x_target_over_alpha_times_anchor_gap"
    else:
        x_next_raw = float(x_target - alpha * (x_target - x_anchor))
        if target_rule == "q_next_anchor":
            x_next_raw = float(x_q50 - alpha * (x_q50 - x_anchor))
        x_next = min(max(x_next_raw, float(grid[0])), float(grid[-1]))
        if x_next >= x_target - 1.0e-8:
            x_next = max(float(grid[0]), float(x_target - grid_dx))
        if x_next >= parent_center - 1.0e-8:
            x_next = max(float(grid[0]), float(parent_center - grid_dx))
        matched_force = float(protocol_k * (x_target - protocol_end_center))
        gap = float(x_target - x_next)
        stiffness_formula = "k_protocol_end_times_x_target_minus_x_protocol_end_over_alpha_times_anchor_gap"
    raw_k = float("nan")
    k_next = float(k_min)
    k_clamped_to = "k_min_fallback_for_nonpositive_force_gap"
    if matched_force > 0.0 and gap > 1.0e-8:
        raw_k = float(matched_force / gap)
        k_next = raw_k
        k_clamped_to = "none"
        if raw_k < k_min:
            k_next = float(k_min)
            k_clamped_to = "k_min"
        elif raw_k > k_max:
            k_next = float(k_max)
            k_clamped_to = "k_max"

    window1_dir = out_root / "window_1"
    run_eq_window(
        bin_path,
        ctx,
        x_next,
        k_next,
        int(ctx.get("mines_screen", {}).get("fixed", {}).get("eq_steps", 100000)),
        int(ctx.get("mines_screen", {}).get("fixed", {}).get("eq_nout", 1000)) * 10,
        seed + seed_offset + 2000,
        window1_dir,
    )
    eq1_rows = read_csv_rows(window1_dir / "eq_window.csv")
    if not eq1_rows:
        raise RuntimeError(f"MiNES child build produced no EQ samples for {child_name}.")
    eq1_fieldnames = list(eq1_rows[0].keys())
    eq1_discard = int(math.floor(len(eq1_rows) * (1.0 - use_tail)))
    eq1_discard = min(max(eq1_discard, 0), max(len(eq1_rows) - 1, 0))
    eq1_tail_rows = eq1_rows[eq1_discard:]
    write_csv(window1_dir / "eq_tail.csv", eq1_fieldnames, eq1_tail_rows)

    final_work = np.asarray([float(row["final_work"]) for row in endpoint_rows], dtype=float)
    hs_mask = np.isfinite(hs_pmf)
    hs_rmse = (
        float(np.sqrt(np.mean((hs_pmf[hs_mask] - analytic_pmf[hs_mask]) ** 2)))
        if np.any(hs_mask)
        else float("nan")
    )

    summary = {
        "label": label,
        "seed": int(seed),
        "parameters": {
            "side": side,
            "parent_name": parent_name,
            "child_name": child_name,
            "k_min": float(k_min),
            "k_max": float(k_max),
            "eq_steps": int(ctx.get("mines_screen", {}).get("fixed", {}).get("eq_steps", 100000)),
            "eq_nout": int(ctx.get("mines_screen", {}).get("fixed", {}).get("eq_nout", 1000)) * 10,
            "T_NES": int(t_nes),
            "N_NES": int(n_nes),
            "use_tail": float(use_tail),
            "alpha": float(alpha),
            "q_next": float(q_next_level),
            "anchor_quantile_level": float(anchor_quantile_level),
            "x0_left": float(ctx["basins"]["left"]),
            "x0_right": float(ctx["basins"]["right"]),
            "x0_parent": float(parent_center),
            "x0_protocol_end": float(protocol_end_center),
            "k0_parent": float(parent_k),
            "k0_protocol_end": float(protocol_k),
            "target_rule": target_rule,
            "next_center_rule": "target_plus_or_minus_alpha_times_anchor_minus_target_by_side",
            "next_k_rule": "force_match_from_reused_constant_endpoint_protocol_k_by_side",
        },
        "window_0": {
            "name": parent_name,
            "center_x": float(parent_center),
            "k": float(parent_k),
            "eq_file": str((window0_dir / "eq_window.csv").relative_to(out_root)),
            "tail_file": str((window0_dir / "eq_tail.csv").relative_to(out_root)),
            "source_eq_file": parent_eq_ref,
            "source_tail_file": parent_tail_ref,
            "n_samples": int(len(parent_eq_rows)),
            "n_tail_samples": int(len(parent_tail_rows)),
        },
        "nes": {
            "draws_file": str((nes_dir / "drawn_start_samples.csv").relative_to(out_root)),
            "endpoints_file": str((nes_dir / "forward_endpoints.csv").relative_to(out_root)),
            "n_forward_traj": int(len(traj_paths)),
            "final_x_q50": float(x_q50),
            "final_x_q_anchor": float(x_anchor),
            "final_x_mean": float(np.mean(final_x)),
            "final_x_std": float(np.std(final_x)),
            "final_work_mean": float(np.mean(final_work)),
            "final_work_std": float(np.std(final_work)),
            "forward_path_example": str(traj_paths[0].relative_to(out_root)),
            "hs_pmf_file": str((out_root / "oneway_hs_pmf.csv").relative_to(out_root)),
            "hs_rmse_vs_analytic": hs_rmse,
        },
        "window_1": {
            "name": child_name,
            "center_x": float(x_next),
            "k_raw": raw_k,
            "k": float(k_next),
            "k_clamped_to": k_clamped_to,
            "target_x": float(x_target),
            "anchor_x": float(x_anchor),
            "anchor_quantile_level": float(anchor_quantile_level),
            "matched_force_at_target": float(matched_force),
            "gap_x_next_minus_target": float(gap),
            "stiffness_formula": stiffness_formula,
            "eq_file": str((window1_dir / "eq_window.csv").relative_to(out_root)),
            "tail_file": str((window1_dir / "eq_tail.csv").relative_to(out_root)),
            "n_samples": int(len(eq1_rows)),
            "n_tail_samples": int(len(eq1_tail_rows)),
        },
    }
    write_json(out_root / "mines_first_try_summary.json", summary)
    return summary


def run_mines_bidirectional_segment_generic(
    system_root: Path,
    ctx: dict,
    seed: int,
    bin_path: str,
    out_root: Path,
    left_name: str,
    right_name: str,
    left_center: float,
    right_center: float,
    left_k: float,
    right_k: float,
    left_tail_rows: list[dict[str, str]],
    right_tail_rows: list[dict[str, str]],
    left_tail_ref: str,
    right_tail_ref: str,
    t_nes: int,
    n_nes: int,
    use_tail: float,
    seed_offset: int,
) -> dict:
    if right_center <= left_center:
        raise RuntimeError(
            f"Expected {right_name} to lie to the right of {left_name}, got {left_center} and {right_center}."
        )
    if not left_tail_rows or not right_tail_rows:
        raise RuntimeError(f"Missing saved tail rows for segment {left_name} <-> {right_name}.")

    out_root.mkdir(parents=True, exist_ok=True)
    grid = build_grid(
        float(ctx["grid"]["xmin"]),
        float(ctx["grid"]["xmax"]),
        float(ctx["grid"]["dx"]),
    )
    analytic_pmf = analytic_doublewell_profile_grid(grid, ctx)

    n_path_rows = t_nes + 2
    fractions = np.linspace(0.0, 1.0, n_path_rows)
    centers = (left_center + fractions * (right_center - left_center)).tolist()
    sqrt_ks = np.sqrt(left_k) + fractions * (np.sqrt(right_k) - np.sqrt(left_k))
    ks = (sqrt_ks * sqrt_ks).tolist()

    protocol_dir = out_root / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    forward_path_file = protocol_dir / "forward_path.csv"
    reverse_path_file = protocol_dir / "reverse_path.csv"
    write_protocol_path(forward_path_file, centers, ks)
    write_protocol_path(reverse_path_file, list(reversed(centers)), list(reversed(ks)))
    forward_path_rows = read_protocol_path_rows(forward_path_file)
    reverse_path_rows = read_protocol_path_rows(reverse_path_file)

    forward_dir = out_root / "forward"
    reverse_dir = out_root / "reverse"
    forward_dir.mkdir(parents=True, exist_ok=True)
    reverse_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + seed_offset)
    draw_count = min(int(n_nes), len(left_tail_rows))
    draw_indices = rng.choice(len(left_tail_rows), size=draw_count, replace=False)
    forward_draw_rows: list[dict] = []
    forward_endpoint_rows: list[dict] = []
    forward_trajectories: list[list[dict[str, str]]] = []

    right_tail_path = forward_dir / "right_tail_reference.csv"
    write_csv(right_tail_path, list(right_tail_rows[0].keys()), right_tail_rows)
    for draw_idx, source_tail_index in enumerate(draw_indices.tolist()):
        source_row = left_tail_rows[int(source_tail_index)]
        start_x = float(source_row["x"])
        start_y = float(source_row.get("y", "0.0") or 0.0)
        forward_draw_rows.append(
            {
                "draw_idx": int(draw_idx),
                "source_tail_index": int(source_tail_index),
                "source_step": int(float(source_row.get("step", "0") or 0.0)),
                "start_x": start_x,
                "start_y": start_y,
                "base_u": float(source_row.get("base_u", "nan") or "nan"),
                "bias_u": float(source_row.get("bias_u", "nan") or "nan"),
            }
        )
        start_path = forward_dir / "starts" / f"start_{draw_idx:03d}.csv"
        write_csv(start_path, ["x", "y"], [{"x": start_x, "y": start_y}])
        traj_dir = forward_dir / f"traj_{draw_idx:03d}"
        run_neq_protocol_single_start(
            bin_path,
            ctx,
            start_path,
            right_tail_path,
            forward_path_file,
            seed + seed_offset + 1000 + draw_idx,
            traj_dir,
        )
        traj_path = traj_dir / "neq_fwd_0.csv"
        traj_rows = read_csv_rows(traj_path)
        if not traj_rows:
            raise RuntimeError(f"MiNES segment {left_name}->{right_name} produced no forward rows for draw {draw_idx}.")
        forward_trajectories.append(traj_rows)
        final_row = traj_rows[-1]
        forward_endpoint_rows.append(
            {
                "draw_idx": int(draw_idx),
                "start_x": start_x,
                "start_y": start_y,
                "final_step": int(float(final_row["step"])),
                "final_lambda": float(final_row["lambda"]),
                "final_x": float(final_row["x"]),
                "final_y": float(final_row.get("y", "0.0") or 0.0),
                "final_work": float(final_row["work"]),
                "traj_file": str(traj_path.relative_to(out_root)),
            }
        )

    write_csv(
        forward_dir / "drawn_start_samples.csv",
        ["draw_idx", "source_tail_index", "source_step", "start_x", "start_y", "base_u", "bias_u"],
        forward_draw_rows,
    )
    write_csv(
        forward_dir / "forward_endpoints.csv",
        ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
        forward_endpoint_rows,
    )

    reverse_draw_count = min(int(n_nes), len(right_tail_rows))
    reverse_draw_indices = rng.choice(len(right_tail_rows), size=reverse_draw_count, replace=False)
    reverse_draw_rows: list[dict] = []
    reverse_endpoint_rows: list[dict] = []
    reverse_trajectories: list[list[dict[str, str]]] = []
    left_tail_path = reverse_dir / "left_tail_reference.csv"
    write_csv(left_tail_path, list(left_tail_rows[0].keys()), left_tail_rows)
    for draw_idx, source_tail_index in enumerate(reverse_draw_indices.tolist()):
        source_row = right_tail_rows[int(source_tail_index)]
        start_x = float(source_row["x"])
        start_y = float(source_row.get("y", "0.0") or 0.0)
        reverse_draw_rows.append(
            {
                "draw_idx": int(draw_idx),
                "source_tail_index": int(source_tail_index),
                "source_step": int(float(source_row.get("step", "0") or 0.0)),
                "start_x": start_x,
                "start_y": start_y,
                "base_u": float(source_row.get("base_u", "nan") or "nan"),
                "bias_u": float(source_row.get("bias_u", "nan") or "nan"),
            }
        )
        start_path = reverse_dir / "starts" / f"start_{draw_idx:03d}.csv"
        write_csv(start_path, ["x", "y"], [{"x": start_x, "y": start_y}])
        traj_dir = reverse_dir / f"traj_{draw_idx:03d}"
        run_neq_protocol_single_start(
            bin_path,
            ctx,
            start_path,
            left_tail_path,
            reverse_path_file,
            seed + seed_offset + 2000 + draw_idx,
            traj_dir,
        )
        traj_path = traj_dir / "neq_fwd_0.csv"
        traj_rows = read_csv_rows(traj_path)
        if not traj_rows:
            raise RuntimeError(f"MiNES segment {right_name}->{left_name} produced no reverse rows for draw {draw_idx}.")
        reverse_trajectories.append(traj_rows)
        final_row = traj_rows[-1]
        reverse_endpoint_rows.append(
            {
                "draw_idx": int(draw_idx),
                "start_x": start_x,
                "start_y": start_y,
                "final_step": int(float(final_row["step"])),
                "final_lambda": float(final_row["lambda"]),
                "final_x": float(final_row["x"]),
                "final_y": float(final_row.get("y", "0.0") or 0.0),
                "final_work": float(final_row["work"]),
                "traj_file": str(traj_path.relative_to(out_root)),
            }
        )

    write_csv(
        reverse_dir / "drawn_start_samples.csv",
        ["draw_idx", "source_tail_index", "source_step", "start_x", "start_y", "base_u", "bias_u"],
        reverse_draw_rows,
    )
    write_csv(
        reverse_dir / "reverse_endpoints.csv",
        ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
        reverse_endpoint_rows,
    )

    forward_hs_pmf = hs_reconstruct_oneway_path_rows(forward_trajectories, forward_path_rows, grid, ctx)
    reverse_hs_pmf = hs_reconstruct_oneway_path_rows(reverse_trajectories, reverse_path_rows, grid, ctx)
    dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    kT = float(ctx["thermal_kT"])
    density_average_pmf = average_pmfs_by_density([forward_hs_pmf, reverse_hs_pmf], dx, kT)
    finite_union = np.isfinite(forward_hs_pmf) | np.isfinite(reverse_hs_pmf)
    density_interval = np.full(len(density_average_pmf), np.nan, dtype=float)
    density_interval[finite_union] = density_average_pmf[finite_union]

    n_time = min(
        len(forward_path_rows),
        min(len(traj) for traj in forward_trajectories) if forward_trajectories else 0,
        min(len(traj) for traj in reverse_trajectories) if reverse_trajectories else 0,
    )
    forward_x = np.asarray(
        [[float(row["x"]) for row in traj[:n_time]] for traj in forward_trajectories],
        dtype=float,
    )
    forward_work_paths = np.asarray(
        [[float(row["work"]) for row in traj[:n_time]] for traj in forward_trajectories],
        dtype=float,
    )
    reverse_x = np.asarray(
        [[float(row["x"]) for row in traj[:n_time]] for traj in reverse_trajectories],
        dtype=float,
    )
    reverse_work_paths = np.asarray(
        [[float(row["work"]) for row in traj[:n_time]] for traj in reverse_trajectories],
        dtype=float,
    )

    forward_hs_pmf_aligned, alignment_shift = align_pmf_to_analytic_anchor(forward_hs_pmf, analytic_pmf, grid, right_center)
    reverse_hs_pmf_aligned, reverse_alignment_shift = align_pmf_to_analytic_anchor(
        reverse_hs_pmf, analytic_pmf, grid, right_center
    )
    alignment_status = f"exact_{right_name.lower()}" if math.isfinite(alignment_shift) else f"{right_name.lower()}_unresolved_no_shift"
    reverse_alignment_status = (
        f"exact_{right_name.lower()}" if math.isfinite(reverse_alignment_shift) else f"{right_name.lower()}_unresolved_no_shift"
    )

    forward_mask = np.isfinite(forward_hs_pmf_aligned)
    reverse_mask = np.isfinite(reverse_hs_pmf_aligned)
    forward_rmse = (
        float(np.sqrt(np.mean((forward_hs_pmf_aligned[forward_mask] - analytic_pmf[forward_mask]) ** 2)))
        if np.any(forward_mask)
        else float("nan")
    )
    reverse_rmse = (
        float(np.sqrt(np.mean((reverse_hs_pmf_aligned[reverse_mask] - analytic_pmf[reverse_mask]) ** 2)))
        if np.any(reverse_mask)
        else float("nan")
    )

    forward_final_x = np.asarray([float(row["final_x"]) for row in forward_endpoint_rows], dtype=float)
    reverse_final_x = np.asarray([float(row["final_x"]) for row in reverse_endpoint_rows], dtype=float)
    forward_works = np.asarray([float(row["final_work"]) for row in forward_endpoint_rows], dtype=float)
    reverse_works = np.asarray([float(row["final_work"]) for row in reverse_endpoint_rows], dtype=float)
    left_tail_x = np.asarray([float(row["x"]) for row in left_tail_rows], dtype=float)
    right_tail_x = np.asarray([float(row["x"]) for row in right_tail_rows], dtype=float)

    pymbar_module = load_pymbar()
    crooks = pymbar_module.other_estimators.bar(
        forward_works,
        reverse_works,
        compute_uncertainty=True,
    )
    if isinstance(crooks, dict):
        delta_f = float(crooks.get("Delta_f", crooks.get("delta_f", float("nan"))))
        delta_f_unc = float(crooks.get("dDelta_f", crooks.get("delta_f_uncertainty", float("nan"))))
    elif isinstance(crooks, tuple):
        delta_f = float(crooks[0])
        delta_f_unc = float(crooks[1]) if len(crooks) > 1 else float("nan")
    else:
        delta_f = float(crooks)
        delta_f_unc = float("nan")

    reverse_prefix_work = reverse_work_paths[:, ::-1][:, :n_time]
    _, _, f_eq23, _, _, _ = estimate_intermediate_reduced_free_energies(
        forward_work_paths[:, :n_time],
        reverse_prefix_work,
        delta_f,
        n_boot=64,
        rng_seed=seed + seed_offset,
    )
    eq23_mts_pmf = build_bidirectional_mts_pmf(
        forward_x[:, :n_time],
        forward_work_paths[:, :n_time],
        reverse_x[:, :n_time],
        reverse_work_paths[:, :n_time],
        np.asarray([float(row["center_x"]) for row in forward_path_rows[:n_time]], dtype=float),
        np.asarray([float(row["k"]) for row in forward_path_rows[:n_time]], dtype=float),
        grid,
        f_eq23[:n_time],
        delta_f,
        kT=kT,
    )

    write_csv(
        out_root / "segment_hs_pmf.csv",
        [
            "x",
            "forward_hs_pmf",
            "forward_hs_pmf_aligned",
            "reverse_hs_pmf",
            "reverse_hs_pmf_aligned",
            "density_average_pmf",
            "eq23_mts_pmf",
            "analytic_pmf",
        ],
        [
            {
                "x": float(x),
                "forward_hs_pmf": "" if not math.isfinite(float(fwd)) else float(fwd),
                "forward_hs_pmf_aligned": "" if not math.isfinite(float(fwd_aligned)) else float(fwd_aligned),
                "reverse_hs_pmf": "" if not math.isfinite(float(rev)) else float(rev),
                "reverse_hs_pmf_aligned": "" if not math.isfinite(float(rev_aligned)) else float(rev_aligned),
                "density_average_pmf": "" if not math.isfinite(float(avg)) else float(avg),
                "eq23_mts_pmf": "" if not math.isfinite(float(mts)) else float(mts),
                "analytic_pmf": float(analytic),
            }
            for x, fwd, fwd_aligned, rev, rev_aligned, avg, mts, analytic in zip(
                grid.tolist(),
                forward_hs_pmf.tolist(),
                forward_hs_pmf_aligned.tolist(),
                reverse_hs_pmf.tolist(),
                reverse_hs_pmf_aligned.tolist(),
                density_interval.tolist(),
                eq23_mts_pmf.tolist(),
                analytic_pmf.tolist(),
            )
        ],
    )

    summary = {
        "left_name": left_name,
        "right_name": right_name,
        "parameters": {
            "T_NES": int(t_nes),
            "N_NES": int(n_nes),
            "use_tail": float(use_tail),
            "path_center_rule": f"linear_{left_name.lower()}_to_{right_name.lower()}",
            "path_k_rule": f"linear_in_sqrt_k_between_{left_name.lower()}_and_{right_name.lower()}",
            "x_left": float(left_center),
            "x_right": float(right_center),
            "k_left": float(left_k),
            "k_right": float(right_k),
        },
        "left": {
            "name": left_name,
            "tail_file": left_tail_ref,
            "center_x": float(left_center),
            "k": float(left_k),
            "n_tail_samples": int(len(left_tail_rows)),
        },
        "right": {
            "name": right_name,
            "tail_file": right_tail_ref,
            "center_x": float(right_center),
            "k": float(right_k),
            "n_tail_samples": int(len(right_tail_rows)),
        },
        "crooks": {
            "delta_f": delta_f,
            "delta_f_uncertainty": delta_f_unc,
            "forward_work_mean": float(np.mean(forward_works)),
            "forward_work_std": float(np.std(forward_works)),
            "reverse_work_mean": float(np.mean(reverse_works)),
            "reverse_work_std": float(np.std(reverse_works)),
        },
        "forward": {
            "draws_file": str((forward_dir / "drawn_start_samples.csv").relative_to(out_root)),
            "endpoints_file": str((forward_dir / "forward_endpoints.csv").relative_to(out_root)),
            "path_file": str(forward_path_file.relative_to(out_root)),
            "n_traj": int(len(forward_trajectories)),
            "final_x_mean": float(np.mean(forward_final_x)),
            "final_x_std": float(np.std(forward_final_x)),
            "final_work_mean": float(np.mean(forward_works)),
            "final_work_std": float(np.std(forward_works)),
            "alignment_anchor_x": float(right_center),
            "alignment_shift": alignment_shift,
            "alignment_status": alignment_status,
            "aligned_hs_rmse_vs_analytic": forward_rmse,
            "right_eq_tail_mean": float(np.mean(right_tail_x)),
            "right_eq_tail_std": float(np.std(right_tail_x)),
        },
        "reverse": {
            "draws_file": str((reverse_dir / "drawn_start_samples.csv").relative_to(out_root)),
            "endpoints_file": str((reverse_dir / "reverse_endpoints.csv").relative_to(out_root)),
            "path_file": str(reverse_path_file.relative_to(out_root)),
            "n_traj": int(len(reverse_trajectories)),
            "final_x_mean": float(np.mean(reverse_final_x)),
            "final_x_std": float(np.std(reverse_final_x)),
            "final_work_mean": float(np.mean(reverse_works)),
            "final_work_std": float(np.std(reverse_works)),
            "alignment_anchor_x": float(right_center),
            "alignment_shift": reverse_alignment_shift,
            "alignment_status": reverse_alignment_status,
            "aligned_hs_rmse_vs_analytic": reverse_rmse,
            "left_eq_tail_mean": float(np.mean(left_tail_x)),
            "left_eq_tail_std": float(np.std(left_tail_x)),
        },
        "pmf": {
            "pmf_file": "segment_hs_pmf.csv",
            "density_average_rmse_vs_analytic": float(
                np.sqrt(
                    np.mean(
                        (density_interval[np.isfinite(density_interval)] - analytic_pmf[np.isfinite(density_interval)]) ** 2
                    )
                )
            )
            if np.any(np.isfinite(density_interval))
            else float("nan"),
            "eq23_mts_rmse_vs_analytic": float(
                np.sqrt(
                    np.mean(
                        (eq23_mts_pmf[np.isfinite(eq23_mts_pmf)] - analytic_pmf[np.isfinite(eq23_mts_pmf)]) ** 2
                    )
                )
            )
            if np.any(np.isfinite(eq23_mts_pmf))
            else float("nan"),
        },
    }
    write_json(out_root / "segment_summary.json", summary)
    return summary


def write_mines_bidirectional_segment_from_saved_paths(
    ctx: dict,
    out_root: Path,
    left_name: str,
    right_name: str,
    left_center: float,
    right_center: float,
    left_k: float,
    right_k: float,
    left_tail_rows: list[dict[str, str]],
    right_tail_rows: list[dict[str, str]],
    left_tail_ref: str,
    right_tail_ref: str,
    forward_path_rows: list[dict[str, str]],
    reverse_path_rows: list[dict[str, str]],
    forward_draw_rows: list[dict],
    forward_endpoint_rows: list[dict],
    forward_trajectories: list[list[dict[str, str]]],
    reverse_draw_rows: list[dict],
    reverse_endpoint_rows: list[dict],
    reverse_trajectories: list[list[dict[str, str]]],
) -> dict:
    if right_center <= left_center:
        raise RuntimeError(
            f"Expected {right_name} to lie to the right of {left_name}, got {left_center} and {right_center}."
        )
    if not left_tail_rows or not right_tail_rows:
        raise RuntimeError(f"Missing saved tail rows for segment {left_name} <-> {right_name}.")

    out_root.mkdir(parents=True, exist_ok=True)
    grid = build_grid(
        float(ctx["grid"]["xmin"]),
        float(ctx["grid"]["xmax"]),
        float(ctx["grid"]["dx"]),
    )
    analytic_pmf = analytic_doublewell_profile_grid(grid, ctx)

    protocol_dir = out_root / "protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    forward_path_file = protocol_dir / "forward_path.csv"
    reverse_path_file = protocol_dir / "reverse_path.csv"
    write_protocol_path(
        forward_path_file,
        [float(row["center_x"]) for row in forward_path_rows],
        [float(row["k"]) for row in forward_path_rows],
    )
    write_protocol_path(
        reverse_path_file,
        [float(row["center_x"]) for row in reverse_path_rows],
        [float(row["k"]) for row in reverse_path_rows],
    )
    forward_path_rows = read_protocol_path_rows(forward_path_file)
    reverse_path_rows = read_protocol_path_rows(reverse_path_file)

    forward_dir = out_root / "forward"
    reverse_dir = out_root / "reverse"
    forward_dir.mkdir(parents=True, exist_ok=True)
    reverse_dir.mkdir(parents=True, exist_ok=True)

    forward_endpoint_rows_out: list[dict] = []
    for draw_idx, (traj_rows, endpoint_row) in enumerate(zip(forward_trajectories, forward_endpoint_rows)):
        traj_dir = forward_dir / f"traj_{draw_idx:03d}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / "neq_fwd_0.csv"
        write_csv(
            traj_path,
            ["step", "lambda", "x", "y", "base_u", "bias_u", "work"],
            traj_rows,
        )
        forward_endpoint_rows_out.append(
            {
                "draw_idx": int(endpoint_row["draw_idx"]),
                "start_x": float(endpoint_row["start_x"]),
                "start_y": float(endpoint_row["start_y"]),
                "final_step": int(endpoint_row["final_step"]),
                "final_lambda": float(endpoint_row["final_lambda"]),
                "final_x": float(endpoint_row["final_x"]),
                "final_y": float(endpoint_row["final_y"]),
                "final_work": float(endpoint_row["final_work"]),
                "traj_file": str(traj_path.relative_to(out_root)),
            }
        )
    write_csv(
        forward_dir / "drawn_start_samples.csv",
        ["draw_idx", "source_tail_index", "source_step", "start_x", "start_y", "base_u", "bias_u"],
        forward_draw_rows,
    )
    write_csv(
        forward_dir / "forward_endpoints.csv",
        ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
        forward_endpoint_rows_out,
    )

    reverse_endpoint_rows_out: list[dict] = []
    for draw_idx, (traj_rows, endpoint_row) in enumerate(zip(reverse_trajectories, reverse_endpoint_rows)):
        traj_dir = reverse_dir / f"traj_{draw_idx:03d}"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / "neq_fwd_0.csv"
        write_csv(
            traj_path,
            ["step", "lambda", "x", "y", "base_u", "bias_u", "work"],
            traj_rows,
        )
        reverse_endpoint_rows_out.append(
            {
                "draw_idx": int(endpoint_row["draw_idx"]),
                "start_x": float(endpoint_row["start_x"]),
                "start_y": float(endpoint_row["start_y"]),
                "final_step": int(endpoint_row["final_step"]),
                "final_lambda": float(endpoint_row["final_lambda"]),
                "final_x": float(endpoint_row["final_x"]),
                "final_y": float(endpoint_row["final_y"]),
                "final_work": float(endpoint_row["final_work"]),
                "traj_file": str(traj_path.relative_to(out_root)),
            }
        )
    write_csv(
        reverse_dir / "drawn_start_samples.csv",
        ["draw_idx", "source_tail_index", "source_step", "start_x", "start_y", "base_u", "bias_u"],
        reverse_draw_rows,
    )
    write_csv(
        reverse_dir / "reverse_endpoints.csv",
        ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
        reverse_endpoint_rows_out,
    )

    forward_hs_pmf = hs_reconstruct_oneway_path_rows(forward_trajectories, forward_path_rows, grid, ctx)
    reverse_hs_pmf = hs_reconstruct_oneway_path_rows(reverse_trajectories, reverse_path_rows, grid, ctx)
    dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    kT = float(ctx["thermal_kT"])
    density_average_pmf = average_pmfs_by_density([forward_hs_pmf, reverse_hs_pmf], dx, kT)
    finite_union = np.isfinite(forward_hs_pmf) | np.isfinite(reverse_hs_pmf)
    density_interval = np.full(len(density_average_pmf), np.nan, dtype=float)
    density_interval[finite_union] = density_average_pmf[finite_union]

    n_time = min(
        len(forward_path_rows),
        min(len(traj) for traj in forward_trajectories) if forward_trajectories else 0,
        min(len(traj) for traj in reverse_trajectories) if reverse_trajectories else 0,
    )
    forward_x = np.asarray(
        [[float(row["x"]) for row in traj[:n_time]] for traj in forward_trajectories],
        dtype=float,
    )
    forward_work_paths = np.asarray(
        [[float(row["work"]) for row in traj[:n_time]] for traj in forward_trajectories],
        dtype=float,
    )
    reverse_x = np.asarray(
        [[float(row["x"]) for row in traj[:n_time]] for traj in reverse_trajectories],
        dtype=float,
    )
    reverse_work_paths = np.asarray(
        [[float(row["work"]) for row in traj[:n_time]] for traj in reverse_trajectories],
        dtype=float,
    )

    forward_hs_pmf_aligned, alignment_shift = align_pmf_to_analytic_anchor(forward_hs_pmf, analytic_pmf, grid, right_center)
    reverse_hs_pmf_aligned, reverse_alignment_shift = align_pmf_to_analytic_anchor(
        reverse_hs_pmf, analytic_pmf, grid, right_center
    )
    alignment_status = f"exact_{right_name.lower()}" if math.isfinite(alignment_shift) else f"{right_name.lower()}_unresolved_no_shift"
    reverse_alignment_status = (
        f"exact_{right_name.lower()}" if math.isfinite(reverse_alignment_shift) else f"{right_name.lower()}_unresolved_no_shift"
    )

    forward_mask = np.isfinite(forward_hs_pmf_aligned)
    reverse_mask = np.isfinite(reverse_hs_pmf_aligned)
    forward_rmse = (
        float(np.sqrt(np.mean((forward_hs_pmf_aligned[forward_mask] - analytic_pmf[forward_mask]) ** 2)))
        if np.any(forward_mask)
        else float("nan")
    )
    reverse_rmse = (
        float(np.sqrt(np.mean((reverse_hs_pmf_aligned[reverse_mask] - analytic_pmf[reverse_mask]) ** 2)))
        if np.any(reverse_mask)
        else float("nan")
    )

    forward_final_x = np.asarray([float(row["final_x"]) for row in forward_endpoint_rows_out], dtype=float)
    reverse_final_x = np.asarray([float(row["final_x"]) for row in reverse_endpoint_rows_out], dtype=float)
    forward_works = np.asarray([float(row["final_work"]) for row in forward_endpoint_rows_out], dtype=float)
    reverse_works = np.asarray([float(row["final_work"]) for row in reverse_endpoint_rows_out], dtype=float)
    left_tail_x = np.asarray([float(row["x"]) for row in left_tail_rows], dtype=float)
    right_tail_x = np.asarray([float(row["x"]) for row in right_tail_rows], dtype=float)

    pymbar_module = load_pymbar()
    crooks = pymbar_module.other_estimators.bar(
        forward_works,
        reverse_works,
        compute_uncertainty=True,
    )
    if isinstance(crooks, dict):
        delta_f = float(crooks.get("Delta_f", crooks.get("delta_f", float("nan"))))
        delta_f_unc = float(crooks.get("dDelta_f", crooks.get("delta_f_uncertainty", float("nan"))))
    elif isinstance(crooks, tuple):
        delta_f = float(crooks[0])
        delta_f_unc = float(crooks[1]) if len(crooks) > 1 else float("nan")
    else:
        delta_f = float(crooks)
        delta_f_unc = float("nan")

    reverse_prefix_work = reverse_work_paths[:, ::-1][:, :n_time]
    _, _, f_eq23, _, _, _ = estimate_intermediate_reduced_free_energies(
        forward_work_paths[:, :n_time],
        reverse_prefix_work,
        delta_f,
        n_boot=64,
        rng_seed=12345,
    )
    eq23_mts_pmf = build_bidirectional_mts_pmf(
        forward_x[:, :n_time],
        forward_work_paths[:, :n_time],
        reverse_x[:, :n_time],
        reverse_work_paths[:, :n_time],
        np.asarray([float(row["center_x"]) for row in forward_path_rows[:n_time]], dtype=float),
        np.asarray([float(row["k"]) for row in forward_path_rows[:n_time]], dtype=float),
        grid,
        f_eq23[:n_time],
        delta_f,
        kT=kT,
    )

    write_csv(
        out_root / "segment_hs_pmf.csv",
        [
            "x",
            "forward_hs_pmf",
            "forward_hs_pmf_aligned",
            "reverse_hs_pmf",
            "reverse_hs_pmf_aligned",
            "density_average_pmf",
            "eq23_mts_pmf",
            "analytic_pmf",
        ],
        [
            {
                "x": float(x),
                "forward_hs_pmf": "" if not math.isfinite(float(fwd)) else float(fwd),
                "forward_hs_pmf_aligned": "" if not math.isfinite(float(fwd_aligned)) else float(fwd_aligned),
                "reverse_hs_pmf": "" if not math.isfinite(float(rev)) else float(rev),
                "reverse_hs_pmf_aligned": "" if not math.isfinite(float(rev_aligned)) else float(rev_aligned),
                "density_average_pmf": "" if not math.isfinite(float(avg)) else float(avg),
                "eq23_mts_pmf": "" if not math.isfinite(float(mts)) else float(mts),
                "analytic_pmf": float(analytic),
            }
            for x, fwd, fwd_aligned, rev, rev_aligned, avg, mts, analytic in zip(
                grid.tolist(),
                forward_hs_pmf.tolist(),
                forward_hs_pmf_aligned.tolist(),
                reverse_hs_pmf.tolist(),
                reverse_hs_pmf_aligned.tolist(),
                density_interval.tolist(),
                eq23_mts_pmf.tolist(),
                analytic_pmf.tolist(),
            )
        ],
    )

    summary = {
        "left_name": left_name,
        "right_name": right_name,
        "parameters": {
            "T_NES": int(max(len(forward_path_rows) - 2, 0)),
            "N_NES": int(min(len(forward_trajectories), len(reverse_trajectories))),
            "use_tail": float("nan"),
            "path_center_rule": f"linear_{left_name.lower()}_to_{right_name.lower()}",
            "path_k_rule": f"linear_in_sqrt_k_between_{left_name.lower()}_and_{right_name.lower()}",
            "x_left": float(left_center),
            "x_right": float(right_center),
            "k_left": float(left_k),
            "k_right": float(right_k),
        },
        "left": {
            "name": left_name,
            "tail_file": left_tail_ref,
            "center_x": float(left_center),
            "k": float(left_k),
            "n_tail_samples": int(len(left_tail_rows)),
        },
        "right": {
            "name": right_name,
            "tail_file": right_tail_ref,
            "center_x": float(right_center),
            "k": float(right_k),
            "n_tail_samples": int(len(right_tail_rows)),
        },
        "crooks": {
            "delta_f": delta_f,
            "delta_f_uncertainty": delta_f_unc,
            "forward_work_mean": float(np.mean(forward_works)),
            "forward_work_std": float(np.std(forward_works)),
            "reverse_work_mean": float(np.mean(reverse_works)),
            "reverse_work_std": float(np.std(reverse_works)),
        },
        "forward": {
            "draws_file": str((forward_dir / "drawn_start_samples.csv").relative_to(out_root)),
            "endpoints_file": str((forward_dir / "forward_endpoints.csv").relative_to(out_root)),
            "path_file": str(forward_path_file.relative_to(out_root)),
            "n_traj": int(len(forward_trajectories)),
            "final_x_mean": float(np.mean(forward_final_x)),
            "final_x_std": float(np.std(forward_final_x)),
            "final_work_mean": float(np.mean(forward_works)),
            "final_work_std": float(np.std(forward_works)),
            "alignment_anchor_x": float(right_center),
            "alignment_shift": alignment_shift,
            "alignment_status": alignment_status,
            "aligned_hs_rmse_vs_analytic": forward_rmse,
            "right_eq_tail_mean": float(np.mean(right_tail_x)),
            "right_eq_tail_std": float(np.std(right_tail_x)),
        },
        "reverse": {
            "draws_file": str((reverse_dir / "drawn_start_samples.csv").relative_to(out_root)),
            "endpoints_file": str((reverse_dir / "reverse_endpoints.csv").relative_to(out_root)),
            "path_file": str(reverse_path_file.relative_to(out_root)),
            "n_traj": int(len(reverse_trajectories)),
            "final_x_mean": float(np.mean(reverse_final_x)),
            "final_x_std": float(np.std(reverse_final_x)),
            "final_work_mean": float(np.mean(reverse_works)),
            "final_work_std": float(np.std(reverse_works)),
            "alignment_anchor_x": float(right_center),
            "alignment_shift": reverse_alignment_shift,
            "alignment_status": reverse_alignment_status,
            "aligned_hs_rmse_vs_analytic": reverse_rmse,
            "left_eq_tail_mean": float(np.mean(left_tail_x)),
            "left_eq_tail_std": float(np.std(left_tail_x)),
        },
        "pmf": {
            "pmf_file": "segment_hs_pmf.csv",
            "density_average_rmse_vs_analytic": float(
                np.sqrt(
                    np.mean(
                        (density_interval[np.isfinite(density_interval)] - analytic_pmf[np.isfinite(density_interval)]) ** 2
                    )
                )
            )
            if np.any(np.isfinite(density_interval))
            else float("nan"),
            "eq23_mts_rmse_vs_analytic": float(
                np.sqrt(
                    np.mean(
                        (eq23_mts_pmf[np.isfinite(eq23_mts_pmf)] - analytic_pmf[np.isfinite(eq23_mts_pmf)]) ** 2
                    )
                )
            )
            if np.any(np.isfinite(eq23_mts_pmf))
            else float("nan"),
        },
    }
    write_json(out_root / "segment_summary.json", summary)
    return summary


def run_aus_seed(system_root: Path, combo_label: str, seed: int, bin_path: str) -> None:
    ctx = load_json(system_root / "run_context.json")
    combo = next(combo for combo in ctx["aus_screen"]["combos"] if combo["label"] == combo_label)
    fixed = ctx["aus_screen"]["fixed"]
    out_root = system_root / "AUS" / combo_label / "raw" / f"seed_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    grid = build_grid(
        float(ctx["grid"]["xmin"]),
        float(ctx["grid"]["xmax"]),
        float(fixed.get("grid_dx", ctx["grid"]["dx"])),
    )
    grid_dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    endpoint_k = float(fixed.get("endpoint_k", 1.0))
    left_start_x = float(fixed.get("start_x_left", ctx["basins"]["left"]))
    right_start_x = float(fixed.get("start_x_right", ctx["basins"]["right"]))
    q_next_level = float(combo["q_next"])
    alpha = float(combo["alpha"])
    fit_method = str(combo.get("fit_method", "poly_4term_parent"))
    k_min = float(combo["k_min"])
    k_max = float(combo["k_max"])
    max_iterations = int(fixed.get("max_iterations", 50))
    rescue_metric = str(fixed.get("refine_metric", "interested_region_fraction_of_average_ess"))
    rescue_half_width = int(fixed.get("refine_half_width_points", 5))
    rescue_k_addition = float(fixed.get("refine_k_addition", 10.0))
    rescue_k_max = float(fixed.get("refine_rescue_k_max", max(k_max, 100.0)))
    eq_steps = int(fixed["eq_steps"])
    eq_nout = int(fixed["eq_nout"])
    time_values = [int(value) for value in ctx.get("time_grid", {}).get("values", [])]
    target_total_steps = int(
        fixed.get(
            "total_steps",
            max(time_values) if time_values else 2 * eq_steps * (max_iterations + 1),
        )
    )
    rescue_ess_min_fraction = float(fixed.get("refine_ess_min_fraction", fixed.get("refine_ess_threshold", 0.05)))
    cumulative = 0
    windows: list[dict] = []
    window_samples: dict[int, list[float]] = {}
    allocated_umbrellas: list[dict[str, int | float | str]] = []

    def record_window(
        *,
        window_id: int,
        side: str,
        depth: int,
        iteration: int,
        parent_window_id: int | str | None,
        center_x: float,
        k: float,
        n_steps: int,
        samples: list[float],
        cumulative_start: int,
        cumulative_end: int,
        iteration_budget_start: int,
        iteration_budget_end: int,
        iteration_window_count: int,
        seed_value: int,
        traj_file: str,
        window_dir: str,
        pair_overlap: float | str | None = "",
        mean_parent: float | str | None = "",
        sigma_parent: float | str | None = "",
        median_parent_x: float | str | None = "",
        q_next_x: float | str | None = "",
        target_mean_x: float | str | None = "",
        local_curvature: float | str | None = "",
        local_slope: float | str | None = "",
        derived_k: float | str | None = "",
        sample_q_lo: float | str | None = "",
        sample_q_hi: float | str | None = "",
        target_in_sample_band: bool | str | None = "",
        k_clamped_to: str = "",
        allocate_unique: bool = False,
    ) -> None:
        windows.append(
            {
                "window_id": int(window_id),
                "side": side,
                "depth": int(depth),
                "iteration": int(iteration),
                "parent_window_id": "" if parent_window_id in ("", None) else parent_window_id,
                "center_x": float(center_x),
                "center_y": 0.0,
                "k": float(k),
                "n_steps": int(n_steps),
                "n_frames": len(samples),
                "cumulative_start": int(cumulative_start),
                "cumulative_end": int(cumulative_end),
                "iteration_budget_start": int(iteration_budget_start),
                "iteration_budget_end": int(iteration_budget_end),
                "iteration_window_count": int(iteration_window_count),
                "pair_overlap": pair_overlap,
                "mean_parent": mean_parent,
                "sigma_parent": sigma_parent,
                "median_parent_x": median_parent_x,
                "q_next_x": q_next_x,
                "target_mean_x": target_mean_x,
                "local_curvature": local_curvature,
                "local_slope": local_slope,
                "derived_k": derived_k,
                "sample_q_lo": sample_q_lo,
                "sample_q_hi": sample_q_hi,
                "target_in_sample_band": target_in_sample_band,
                "k_clamped_to": k_clamped_to,
                "alpha": alpha,
                "fit_method": fit_method,
                "q_next": q_next_level,
                "k_min": k_min,
                "k_max": k_max,
                "seed": int(seed_value),
                "traj_file": traj_file,
                "window_dir": window_dir,
            }
        )
        window_samples[int(window_id)] = list(samples)
        if allocate_unique:
            allocated_umbrellas.append(
                {
                    "window_id": int(window_id),
                    "side": side,
                    "depth": int(depth),
                    "center_x": float(center_x),
                    "k": float(k),
                }
            )

    initial_pair_steps = min(eq_steps, target_total_steps // 2 if target_total_steps > 0 else eq_steps)
    if initial_pair_steps <= 0:
        raise RuntimeError("AUS total budget must allow at least one step per endpoint window.")
    left_samples, left_file = run_eq_window(
        bin_path, ctx, left_start_x, endpoint_k, initial_pair_steps, eq_nout, seed + 11, out_root / "window_0",
    )
    record_window(
        window_id=0,
        side="left",
        depth=0,
        iteration=0,
        parent_window_id="",
        center_x=left_start_x,
        k=endpoint_k,
        n_steps=initial_pair_steps,
        samples=left_samples,
        cumulative_start=cumulative,
        cumulative_end=cumulative + initial_pair_steps,
        iteration_budget_start=0,
        iteration_budget_end=2 * initial_pair_steps,
        iteration_window_count=2,
        seed_value=seed + 11,
        traj_file=left_file,
        window_dir="window_0",
        allocate_unique=True,
    )
    cumulative += initial_pair_steps
    right_samples, right_file = run_eq_window(
        bin_path, ctx, right_start_x, endpoint_k, initial_pair_steps, eq_nout, seed + 29, out_root / "window_1",
    )
    record_window(
        window_id=1,
        side="right",
        depth=0,
        iteration=0,
        parent_window_id="",
        center_x=right_start_x,
        k=endpoint_k,
        n_steps=initial_pair_steps,
        samples=right_samples,
        cumulative_start=cumulative,
        cumulative_end=cumulative + initial_pair_steps,
        iteration_budget_start=0,
        iteration_budget_end=2 * initial_pair_steps,
        iteration_window_count=2,
        seed_value=seed + 29,
        traj_file=right_file,
        window_dir="window_1",
        allocate_unique=True,
    )
    cumulative += initial_pair_steps

    left_frontier = {"window_id": 0, "center": left_start_x, "samples": left_samples, "depth": 0, "k": endpoint_k}
    right_frontier = {"window_id": 1, "center": right_start_x, "samples": right_samples, "depth": 0, "k": endpoint_k}
    initial_overlap = overlap_coefficient(left_samples, right_samples, grid)
    windows[0]["pair_overlap"] = initial_overlap
    windows[1]["pair_overlap"] = initial_overlap

    next_window_id = 2
    next_seed = seed + 100
    stop_reason = "max_depth"
    for depth in range(1, max_iterations + 1):
            left_plan, left_error = plan_aus_child(
                left_frontier,
                "left",
                grid,
                grid_dx,
                q_next_level,
                alpha,
                fit_method,
                k_min,
                k_max,
                ctx,
            )
            if left_plan is None:
                stop_reason = str(left_error)
                break
            right_plan, right_error = plan_aus_child(
                right_frontier,
                "right",
                grid,
                grid_dx,
                q_next_level,
                alpha,
                fit_method,
                k_min,
                k_max,
                ctx,
            )
            if right_plan is None:
                stop_reason = str(right_error)
                break

            if float(left_plan["q_next_x"]) > float(right_plan["q_next_x"]):
                stop_reason = "quantile_crossing"
                break

            remaining_pair_budget = max(0, target_total_steps - cumulative)
            pair_steps = min(eq_steps, remaining_pair_budget // 2)
            if pair_steps <= 0:
                stop_reason = "budget_exhausted_before_match"
                break
            iteration_budget_start = cumulative
            iteration_budget_end = cumulative + 2 * pair_steps

            left_window_id = next_window_id
            left_seed = next_seed
            left_window_dir = out_root / f"window_{left_window_id}"
            left_child_samples, left_traj_file = run_eq_window(
                bin_path, ctx, float(left_plan["center_x"]), float(left_plan["k"]),
                pair_steps, eq_nout, left_seed, left_window_dir,
            )
            left_start = cumulative
            record_window(
                window_id=left_window_id,
                side="left",
                depth=int(left_plan["depth"]),
                iteration=int(left_plan["iteration"]),
                parent_window_id=int(left_plan["parent_window_id"]),
                center_x=float(left_plan["center_x"]),
                k=float(left_plan["k"]),
                n_steps=pair_steps,
                samples=left_child_samples,
                cumulative_start=left_start,
                cumulative_end=left_start + pair_steps,
                iteration_budget_start=iteration_budget_start,
                iteration_budget_end=iteration_budget_end,
                iteration_window_count=2,
                seed_value=left_seed,
                traj_file=left_traj_file,
                window_dir=f"window_{left_window_id}",
                mean_parent=left_plan["mean_parent"],
                sigma_parent=left_plan["sigma_parent"],
                median_parent_x=left_plan["median_parent_x"],
                q_next_x=left_plan["q_next_x"],
                target_mean_x=left_plan["target_mean_x"],
                local_slope=left_plan["local_slope"],
                derived_k=left_plan["derived_k"],
                k_clamped_to=str(left_plan["k_clamped_to"]),
                allocate_unique=True,
            )
            cumulative += pair_steps
            next_window_id += 1
            next_seed += 1

            right_window_id = next_window_id
            right_seed = next_seed
            right_window_dir = out_root / f"window_{right_window_id}"
            right_child_samples, right_traj_file = run_eq_window(
                bin_path, ctx, float(right_plan["center_x"]), float(right_plan["k"]),
                pair_steps, eq_nout, right_seed, right_window_dir,
            )
            right_start = cumulative
            record_window(
                window_id=right_window_id,
                side="right",
                depth=int(right_plan["depth"]),
                iteration=int(right_plan["iteration"]),
                parent_window_id=int(right_plan["parent_window_id"]),
                center_x=float(right_plan["center_x"]),
                k=float(right_plan["k"]),
                n_steps=pair_steps,
                samples=right_child_samples,
                cumulative_start=right_start,
                cumulative_end=right_start + pair_steps,
                iteration_budget_start=iteration_budget_start,
                iteration_budget_end=iteration_budget_end,
                iteration_window_count=2,
                seed_value=right_seed,
                traj_file=right_traj_file,
                window_dir=f"window_{right_window_id}",
                mean_parent=right_plan["mean_parent"],
                sigma_parent=right_plan["sigma_parent"],
                median_parent_x=right_plan["median_parent_x"],
                q_next_x=right_plan["q_next_x"],
                target_mean_x=right_plan["target_mean_x"],
                local_slope=right_plan["local_slope"],
                derived_k=right_plan["derived_k"],
                k_clamped_to=str(right_plan["k_clamped_to"]),
                allocate_unique=True,
            )
            cumulative += pair_steps
            next_window_id += 1
            next_seed += 1

            pair_overlap = overlap_coefficient(left_child_samples, right_child_samples, grid)
            windows[-2]["pair_overlap"] = pair_overlap
            windows[-1]["pair_overlap"] = pair_overlap

            left_frontier = {
                "window_id": left_window_id,
                "center": float(left_plan["center_x"]),
                "samples": left_child_samples,
                "depth": int(left_plan["depth"]),
                "k": float(left_plan["k"]),
            }
            right_frontier = {
                "window_id": right_window_id,
                "center": float(right_plan["center_x"]),
                "samples": right_child_samples,
                "depth": int(right_plan["depth"]),
                "k": float(right_plan["k"]),
            }

    rescue_summary: dict[str, float | int | str | bool | None] = {
        "refine_metric": rescue_metric,
        "refine_half_width_points": rescue_half_width,
        "refine_k_addition": rescue_k_addition,
        "refine_rescue_k_max": rescue_k_max,
        "refine_ess_min_fraction": rescue_ess_min_fraction,
        "curvature_fit_method": (
            "poly_4term_target_bin_local"
            if fit_method == "poly_4term_parent"
            else "cubic_spline_local"
        ),
        "target_total_steps": target_total_steps,
        "interested_region_left_x": left_start_x,
        "interested_region_right_x": right_start_x,
        "refine_window_added": False,
        "refine_window_extended": False,
        "refine_rounds": 0,
        "low_ess_x": None,
        "low_ess_value": None,
        "low_ess_fraction_of_average": None,
        "region_average_ess": None,
        "region_ess_threshold": None,
        "interested_region_candidate_count": 0,
        "negative_curvature_x": None,
        "negative_curvature": None,
        "local_curvature": None,
        "local_slope": None,
        "local_fit_mode": None,
        "refine_window_id": None,
        "refine_window_center_x": None,
        "refine_window_k": None,
        "refine_extension_window_id": None,
        "refine_extension_parent_window_id": None,
        "refine_extension_center_x": None,
        "refine_extension_k": None,
        "refine_extension_fraction": None,
        "refine_action": None,
        "all_resolved_ess_above_threshold": False,
        "all_region_ess_above_fractional_threshold": False,
        "redistribution_rounds": 0,
        "redistributed_window_runs": 0,
        "current_sampling_cycle": 1,
        "max_sampling_cycle": 1,
        "allocated_umbrella_count": len(allocated_umbrellas),
        "unresolvable_target_count": 0,
        "last_unresolvable_target_x": None,
        "last_refine_sample_q_lo": None,
        "last_refine_sample_q_hi": None,
        "last_refine_target_in_sample_band": None,
        "unused_budget_steps": max(target_total_steps - cumulative, 0),
    }

    if stop_reason == "quantile_crossing" and cumulative < target_total_steps:
        keep_fraction = float(fixed.get("analysis_tail_fraction", 1.0))
        decision_max_samples = int(fixed.get("decision_max_samples_per_window", 1000))
        post_match_iteration = max(int(window["iteration"]) for window in windows) + 1
        current_sampling_cycle = 1
        post_match_initial_f: np.ndarray | None = None
        unresolvable_low_ess_indices: set[int] = set()

        def reconstruct_post_match_state() -> tuple[np.ndarray, np.ndarray]:
            nonlocal post_match_initial_f
            ordered_windows = sorted(windows, key=lambda row: int(row["window_id"]))
            tail_samples = [
                np.asarray(
                    downsample_ordered_values(
                        tail_fraction_values(window_samples[int(window["window_id"])], keep_fraction),
                        decision_max_samples,
                    ),
                    dtype=float,
                )
                for window in ordered_windows
            ]
            tail_centers = [float(window["center_x"]) for window in ordered_windows]
            tail_ks = [float(window["k"]) for window in ordered_windows]
            free_energy, post_match_initial_f, _, ess = us_mbar_profile_details(
                tail_samples,
                tail_centers,
                tail_ks,
                grid,
                ctx,
                post_match_initial_f,
            )
            return free_energy, ess

        def append_rescue_window(low_ess_idx: int, rescue_steps: int) -> None:
            nonlocal cumulative, next_window_id, next_seed, post_match_iteration

            low_ess_x = float(grid[low_ess_idx])
            curvature_probe = local_spline_curvature_probe(
                grid,
                free_energy,
                low_ess_idx,
                rescue_half_width,
            )
            refine_center_x = low_ess_x
            refine_target_x = low_ess_x
            refine_k_raw = float(rescue_k_addition)
            refine_reason = "refine_low_ess_fallback"
            rescue_summary["negative_curvature_x"] = None
            rescue_summary["negative_curvature"] = None
            if curvature_probe is not None:
                rescue_summary["negative_curvature_x"] = float(curvature_probe["negative_curvature_x"])
                rescue_summary["negative_curvature"] = float(curvature_probe["negative_curvature"])
                if float(curvature_probe["negative_curvature"]) < 0.0:
                    refine_center_x = nearest_grid_value(float(curvature_probe["negative_curvature_x"]), grid)
                    refine_target_x = float(curvature_probe["negative_curvature_x"])
                    refine_k_raw = float(-float(curvature_probe["negative_curvature"]) + rescue_k_addition)
                    refine_reason = "refine_negative_curvature"

            refine_k = min(max(refine_k_raw, k_min), k_max)
            refine_clamp = "none"
            if refine_k_raw > k_max:
                refine_clamp = "k_max"
            elif refine_k_raw < k_min:
                refine_clamp = "k_min"

            refine_window_id = next_window_id
            refine_seed = next_seed
            refine_window_dir = out_root / f"window_{refine_window_id}"
            refine_samples, refine_traj_file = run_eq_window(
                bin_path,
                ctx,
                float(refine_center_x),
                float(refine_k),
                rescue_steps,
                eq_nout,
                refine_seed,
                refine_window_dir,
            )
            record_window(
                window_id=refine_window_id,
                side="refine",
                depth=post_match_iteration,
                iteration=post_match_iteration,
                parent_window_id="",
                center_x=float(refine_center_x),
                k=float(refine_k),
                n_steps=rescue_steps,
                samples=refine_samples,
                cumulative_start=cumulative,
                cumulative_end=cumulative + rescue_steps,
                iteration_budget_start=cumulative,
                iteration_budget_end=cumulative + rescue_steps,
                iteration_window_count=1,
                seed_value=refine_seed,
                traj_file=refine_traj_file,
                window_dir=f"window_{refine_window_id}",
                q_next_x=low_ess_x,
                target_mean_x=refine_target_x,
                derived_k=refine_k_raw,
                k_clamped_to=f"{refine_reason}__cycle_{current_sampling_cycle}__{refine_clamp}",
                allocate_unique=True,
            )
            cumulative += rescue_steps
            next_window_id += 1
            next_seed += 1
            post_match_iteration += 1
            rescue_summary["refine_window_added"] = True
            rescue_summary["refine_action"] = "new_refine_window"
            rescue_summary["refine_rounds"] = int(rescue_summary["refine_rounds"]) + 1
            rescue_summary["refine_window_id"] = int(refine_window_id)
            rescue_summary["refine_window_center_x"] = float(refine_center_x)
            rescue_summary["refine_window_k"] = float(refine_k)
            rescue_summary["refine_extension_window_id"] = None
            rescue_summary["refine_extension_parent_window_id"] = None
            rescue_summary["refine_extension_center_x"] = None
            rescue_summary["refine_extension_k"] = None
            rescue_summary["refine_extension_fraction"] = None
            rescue_summary["allocated_umbrella_count"] = len(allocated_umbrellas)
            rescue_summary["current_sampling_cycle"] = int(current_sampling_cycle)
            rescue_summary["max_sampling_cycle"] = max(int(rescue_summary["max_sampling_cycle"]), int(current_sampling_cycle))

        def append_poly_target_rescue_window(low_ess_idx: int, rescue_steps: int) -> None:
            nonlocal cumulative, next_window_id, next_seed, post_match_iteration

            low_ess_x = float(grid[low_ess_idx])
            fit_probe = local_poly_4term_rescue_probe(
                grid,
                free_energy,
                low_ess_idx,
                rescue_half_width,
                flank_points=5,
            )
            fit_slope = 0.0
            fit_curvature = 0.0
            fit_mode = "unresolved_target_fallback"
            if fit_probe is not None:
                fit_slope = float(fit_probe["slope"])
                fit_curvature = float(fit_probe["curvature"])
                fit_mode = str(fit_probe["fit_mode"])

            rescue_summary["negative_curvature_x"] = None
            rescue_summary["negative_curvature"] = None
            rescue_summary["local_curvature"] = fit_curvature if math.isfinite(fit_curvature) else None
            rescue_summary["local_slope"] = fit_slope if math.isfinite(fit_slope) else None
            rescue_summary["local_fit_mode"] = fit_mode
            if math.isfinite(fit_curvature) and fit_curvature < 0.0:
                rescue_summary["negative_curvature_x"] = low_ess_x
                rescue_summary["negative_curvature"] = fit_curvature

            refine_center_x = nearest_grid_value(low_ess_x, grid)
            refine_target_x = low_ess_x
            refine_k_raw = (
                float(abs(fit_curvature) + rescue_k_addition)
                if math.isfinite(fit_curvature)
                else float(rescue_k_addition)
            )
            refine_reason = (
                "refine_abs_curvature_target_bin_fallback"
                if fit_mode == "unresolved_target_fallback"
                else
                "refine_abs_curvature_target_bin_poly3_bridge"
                if fit_mode == "bridge_from_first_resolved_left_right_poly3"
                else "refine_abs_curvature_target_bin_poly4"
            )

            while cumulative < target_total_steps:
                current_steps = min(rescue_steps, target_total_steps - cumulative)
                if current_steps <= 0:
                    return

                refine_k = min(max(refine_k_raw, k_min), rescue_k_max)
                refine_clamp = "none"
                if refine_k_raw > rescue_k_max:
                    refine_clamp = "k_rescue_max"
                elif refine_k_raw < k_min:
                    refine_clamp = "k_min"

                refine_window_id = next_window_id
                refine_seed = next_seed
                refine_window_dir = out_root / f"window_{refine_window_id}"
                refine_samples, refine_traj_file = run_eq_window(
                    bin_path,
                    ctx,
                    float(refine_center_x),
                    float(refine_k),
                    current_steps,
                    eq_nout,
                    refine_seed,
                    refine_window_dir,
                )
                sample_band = quantile_band(refine_samples, q_next_level)
                sample_q_lo = None if sample_band is None else float(sample_band[0])
                sample_q_hi = None if sample_band is None else float(sample_band[1])
                target_in_sample_band = bool(
                    sample_band is not None
                    and float(sample_band[0]) <= low_ess_x <= float(sample_band[1])
                )

                record_window(
                    window_id=refine_window_id,
                    side="refine",
                    depth=post_match_iteration,
                    iteration=post_match_iteration,
                    parent_window_id="",
                    center_x=float(refine_center_x),
                    k=float(refine_k),
                    n_steps=current_steps,
                    samples=refine_samples,
                    cumulative_start=cumulative,
                    cumulative_end=cumulative + current_steps,
                    iteration_budget_start=cumulative,
                    iteration_budget_end=cumulative + current_steps,
                    iteration_window_count=1,
                    seed_value=refine_seed,
                    traj_file=refine_traj_file,
                    window_dir=f"window_{refine_window_id}",
                    q_next_x=low_ess_x,
                    target_mean_x=refine_target_x,
                    local_slope=fit_slope,
                    local_curvature=fit_curvature,
                    derived_k=refine_k_raw,
                    sample_q_lo=sample_q_lo,
                    sample_q_hi=sample_q_hi,
                    target_in_sample_band=target_in_sample_band,
                    k_clamped_to=f"{refine_reason}__cycle_{current_sampling_cycle}__{refine_clamp}",
                    allocate_unique=True,
                )
                cumulative += current_steps
                next_window_id += 1
                next_seed += 1
                post_match_iteration += 1
                rescue_summary["refine_window_added"] = True
                rescue_summary["refine_rounds"] = int(rescue_summary["refine_rounds"]) + 1
                rescue_summary["refine_window_id"] = int(refine_window_id)
                rescue_summary["refine_window_center_x"] = float(refine_center_x)
                rescue_summary["refine_window_k"] = float(refine_k)
                rescue_summary["refine_extension_window_id"] = None
                rescue_summary["refine_extension_parent_window_id"] = None
                rescue_summary["refine_extension_center_x"] = None
                rescue_summary["refine_extension_k"] = None
                rescue_summary["refine_extension_fraction"] = None
                rescue_summary["allocated_umbrella_count"] = len(allocated_umbrellas)
                rescue_summary["current_sampling_cycle"] = int(current_sampling_cycle)
                rescue_summary["max_sampling_cycle"] = max(int(rescue_summary["max_sampling_cycle"]), int(current_sampling_cycle))
                rescue_summary["last_refine_sample_q_lo"] = sample_q_lo
                rescue_summary["last_refine_sample_q_hi"] = sample_q_hi
                rescue_summary["last_refine_target_in_sample_band"] = bool(target_in_sample_band)

                if target_in_sample_band:
                    rescue_summary["refine_action"] = "new_refine_window"
                    return

                if refine_k >= rescue_k_max - 1.0e-12:
                    unresolvable_low_ess_indices.add(int(low_ess_idx))
                    rescue_summary["refine_action"] = "mark_unresolvable_target"
                    rescue_summary["unresolvable_target_count"] = int(len(unresolvable_low_ess_indices))
                    rescue_summary["last_unresolvable_target_x"] = low_ess_x
                    return

                refine_k_raw = max(refine_k_raw * 2.0, refine_k * 2.0)

        def extend_existing_window(extension_candidate: dict, low_ess_idx: int, rescue_steps: int) -> None:
            nonlocal cumulative, next_window_id, next_seed, post_match_iteration

            extend_window_id = next_window_id
            extend_seed = next_seed
            extend_window_dir = out_root / f"window_{extend_window_id}"
            extend_center_x = float(extension_candidate["center_x"])
            extend_k = float(extension_candidate["k"])
            extend_samples, extend_traj_file = run_eq_window(
                bin_path,
                ctx,
                extend_center_x,
                extend_k,
                rescue_steps,
                eq_nout,
                extend_seed,
                extend_window_dir,
            )
            record_window(
                window_id=extend_window_id,
                side="refine_extend",
                depth=post_match_iteration,
                iteration=post_match_iteration,
                parent_window_id=int(extension_candidate["prototype_window_id"]),
                center_x=extend_center_x,
                k=extend_k,
                n_steps=rescue_steps,
                samples=extend_samples,
                cumulative_start=cumulative,
                cumulative_end=cumulative + rescue_steps,
                iteration_budget_start=cumulative,
                iteration_budget_end=cumulative + rescue_steps,
                iteration_window_count=1,
                seed_value=extend_seed,
                traj_file=extend_traj_file,
                window_dir=f"window_{extend_window_id}",
                q_next_x=float(grid[low_ess_idx]),
                target_mean_x=float(grid[low_ess_idx]),
                derived_k=extend_k,
                k_clamped_to=f"refine_extend_existing_window__cycle_{current_sampling_cycle}",
            )
            cumulative += rescue_steps
            next_window_id += 1
            next_seed += 1
            post_match_iteration += 1
            rescue_summary["refine_window_extended"] = True
            rescue_summary["refine_action"] = "extend_existing_window"
            rescue_summary["refine_rounds"] = int(rescue_summary["refine_rounds"]) + 1
            rescue_summary["refine_window_id"] = None
            rescue_summary["refine_window_center_x"] = None
            rescue_summary["refine_window_k"] = None
            rescue_summary["negative_curvature_x"] = None
            rescue_summary["negative_curvature"] = None
            rescue_summary["refine_extension_window_id"] = int(extend_window_id)
            rescue_summary["refine_extension_parent_window_id"] = int(extension_candidate["prototype_window_id"])
            rescue_summary["refine_extension_center_x"] = extend_center_x
            rescue_summary["refine_extension_k"] = extend_k
            rescue_summary["refine_extension_fraction"] = float(extension_candidate["sample_fraction"])
            rescue_summary["allocated_umbrella_count"] = len(allocated_umbrellas)
            rescue_summary["current_sampling_cycle"] = int(current_sampling_cycle)
            rescue_summary["max_sampling_cycle"] = max(int(rescue_summary["max_sampling_cycle"]), int(current_sampling_cycle))

        while cumulative < target_total_steps:
            free_energy, ess = reconstruct_post_match_state()
            ess_state = interested_region_ess_state(
                grid,
                free_energy,
                ess,
                left_start_x,
                right_start_x,
                rescue_half_width,
                rescue_ess_min_fraction,
                excluded_indices=unresolvable_low_ess_indices,
            )
            rescue_summary["current_sampling_cycle"] = int(current_sampling_cycle)
            rescue_summary["max_sampling_cycle"] = max(int(rescue_summary["max_sampling_cycle"]), int(current_sampling_cycle))
            if ess_state is None:
                stop_reason = (
                    "quantile_crossing_all_low_ess_targets_unresolvable"
                    if unresolvable_low_ess_indices
                    else "quantile_crossing_no_resolved_ess_target"
                )
                break
            low_ess_idx = int(ess_state["low_idx"])
            rescue_summary["low_ess_x"] = float(grid[low_ess_idx])
            rescue_summary["low_ess_value"] = float(ess_state["low_ess_value"])
            rescue_summary["low_ess_fraction_of_average"] = float(ess_state["low_fraction_of_average"])
            rescue_summary["region_average_ess"] = float(ess_state["average_ess"])
            rescue_summary["region_ess_threshold"] = float(ess_state["threshold"])
            rescue_summary["interested_region_candidate_count"] = int(len(ess_state["candidate_indices"]))
            rescue_summary["all_resolved_ess_above_threshold"] = bool(ess_state["all_above_threshold"])
            rescue_summary["all_region_ess_above_fractional_threshold"] = bool(ess_state["all_above_threshold"])
            remaining_budget = target_total_steps - cumulative
            if not rescue_summary["all_resolved_ess_above_threshold"]:
                rescue_steps = min(eq_steps * current_sampling_cycle, remaining_budget)
                if rescue_steps <= 0:
                    break
                if fit_method == "poly_4term_parent":
                    append_poly_target_rescue_window(low_ess_idx, rescue_steps)
                else:
                    extension_candidate = choose_extension_window(
                        allocated_umbrellas,
                        windows,
                        window_samples,
                        grid,
                        low_ess_idx,
                        rescue_ess_min_fraction,
                    )
                    if extension_candidate is not None:
                        extend_existing_window(extension_candidate, low_ess_idx, rescue_steps)
                    else:
                        append_rescue_window(low_ess_idx, rescue_steps)
                continue

            prototypes = list(allocated_umbrellas)
            active_count = len(prototypes)
            if active_count <= 0:
                stop_reason = "quantile_crossing_no_allocated_umbrellas"
                break
            if remaining_budget <= 0:
                break

            next_cycle = current_sampling_cycle + 1
            if remaining_budget >= active_count * eq_steps:
                round_steps = eq_steps
            else:
                round_steps = remaining_budget // active_count
                if round_steps <= 0:
                    active_count = min(active_count, remaining_budget)
                    prototypes = prototypes[:active_count]
                    round_steps = 1

            iteration_budget_start = cumulative
            iteration_budget_end = cumulative + active_count * round_steps
            for prototype in prototypes:
                redistribute_window_id = next_window_id
                redistribute_seed = next_seed
                redistribute_window_dir = out_root / f"window_{redistribute_window_id}"
                redistribute_samples, redistribute_traj_file = run_eq_window(
                    bin_path,
                    ctx,
                    float(prototype["center_x"]),
                    float(prototype["k"]),
                    round_steps,
                    eq_nout,
                    redistribute_seed,
                    redistribute_window_dir,
                )
                record_window(
                    window_id=redistribute_window_id,
                    side="redistribute",
                    depth=post_match_iteration,
                    iteration=post_match_iteration,
                    parent_window_id=int(prototype["window_id"]),
                    center_x=float(prototype["center_x"]),
                    k=float(prototype["k"]),
                    n_steps=round_steps,
                    samples=redistribute_samples,
                    cumulative_start=cumulative,
                    cumulative_end=cumulative + round_steps,
                    iteration_budget_start=iteration_budget_start,
                    iteration_budget_end=iteration_budget_end,
                    iteration_window_count=active_count,
                    seed_value=redistribute_seed,
                    traj_file=redistribute_traj_file,
                    window_dir=f"window_{redistribute_window_id}",
                    k_clamped_to=f"redistribute_existing_window__cycle_{next_cycle}",
                )
                cumulative += round_steps
                next_window_id += 1
                next_seed += 1
            post_match_iteration += 1
            rescue_summary["redistribution_rounds"] = int(rescue_summary["redistribution_rounds"]) + 1
            rescue_summary["redistributed_window_runs"] = int(rescue_summary["redistributed_window_runs"]) + active_count
            if round_steps == eq_steps and active_count == len(allocated_umbrellas):
                current_sampling_cycle = next_cycle
            rescue_summary["current_sampling_cycle"] = int(current_sampling_cycle)
            rescue_summary["max_sampling_cycle"] = max(int(rescue_summary["max_sampling_cycle"]), int(current_sampling_cycle))

        if stop_reason == "quantile_crossing":
            if int(rescue_summary["redistribution_rounds"]) > 0:
                stop_reason = "quantile_crossing_with_iterative_budget_cycles"
            elif int(rescue_summary["refine_rounds"]) > 0 and cumulative >= target_total_steps:
                stop_reason = "quantile_crossing_with_refine_budget_exhausted"
            elif rescue_summary["all_resolved_ess_above_threshold"]:
                stop_reason = "quantile_crossing_with_ess_threshold"
            elif int(rescue_summary["refine_rounds"]) > 0:
                stop_reason = "quantile_crossing_with_refine"

    rescue_summary["allocated_umbrella_count"] = len(allocated_umbrellas)
    rescue_summary["unused_budget_steps"] = max(target_total_steps - cumulative, 0)

    write_csv(
        out_root / "aus_windows.csv",
        [
            "window_id", "side", "depth", "iteration", "parent_window_id",
            "center_x", "center_y", "k", "n_steps", "n_frames",
            "cumulative_start", "cumulative_end", "iteration_budget_start",
            "iteration_budget_end", "iteration_window_count", "pair_overlap", "mean_parent",
            "sigma_parent", "median_parent_x", "q_next_x", "target_mean_x",
            "local_curvature",
            "local_slope", "derived_k", "sample_q_lo", "sample_q_hi",
            "target_in_sample_band", "k_clamped_to", "alpha", "fit_method", "q_next",
            "k_min", "k_max", "seed", "traj_file", "window_dir",
        ],
        windows,
    )
    write_json(
        out_root / "aus_summary.json",
        {
            "combo_label": combo_label,
            "seed": seed,
            "alpha": alpha,
            "fit_method": fit_method,
            "q_next": q_next_level,
            "endpoint_k": endpoint_k,
            "k_min": k_min,
            "k_max": k_max,
            "eq_steps": fixed["eq_steps"],
            "eq_nout": fixed["eq_nout"],
            "analysis_tail_fraction": fixed.get("analysis_tail_fraction", 1.0),
            "target_total_steps": target_total_steps,
            "grid_dx": fixed.get("grid_dx", ctx["grid"]["dx"]),
            "start_x_left": left_start_x,
            "start_x_right": right_start_x,
            "max_iterations": max_iterations,
            "stop_reason": stop_reason,
            "n_windows": len(windows),
            "n_left_windows": sum(1 for row in windows if row["side"] == "left"),
            "n_right_windows": sum(1 for row in windows if row["side"] == "right"),
            "n_refine_windows": sum(1 for row in windows if row["side"] == "refine"),
            "n_refine_extend_windows": sum(1 for row in windows if row["side"] == "refine_extend"),
            "n_redistribute_windows": sum(1 for row in windows if row["side"] == "redistribute"),
            "total_steps": cumulative,
            "resolved_overlap": bool(float(np.quantile(np.asarray(left_frontier["samples"], dtype=float), q_next_level)) > float(np.quantile(np.asarray(right_frontier["samples"], dtype=float), 1.0 - q_next_level))),
            "final_pair_overlap": next((float(row["pair_overlap"]) for row in reversed(windows) if row["pair_overlap"] != ""), None),
            "final_left_frontier_x": float(left_frontier["center"]),
            "final_right_frontier_x": float(right_frontier["center"]),
            "final_left_q_next_x": float(np.quantile(np.asarray(left_frontier["samples"], dtype=float), q_next_level)),
            "final_right_q_prev_x": float(np.quantile(np.asarray(right_frontier["samples"], dtype=float), 1.0 - q_next_level)),
            **rescue_summary,
        },
    )


def run_mines_seed(system_root: Path, combo_label: str, seed: int, bin_path: str) -> None:
    ctx = load_json(system_root / "run_context.json")
    combo = next(combo for combo in ctx["mines_screen"]["combos"] if combo["label"] == combo_label)
    fixed = ctx["mines_screen"]["fixed"]
    out_root = system_root / "MINES" / combo_label / "raw" / f"seed_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    grid = build_grid(
        float(ctx["rmse_eval_grid"]["xmin"]),
        float(ctx["rmse_eval_grid"]["xmax"]),
        float(fixed.get("grid_dx", ctx["rmse_eval_grid"]["dx"])),
    )
    beta = 1.0 / float(ctx["thermal_kT"])
    left_center = float(ctx["basins"]["left"])
    right_center = float(ctx["basins"]["right"])
    cumulative = 0
    next_seed = seed + 1000

    milestones: list[dict] = []
    left_points = []
    right_points = []

    # MINES starts from endpoint milestones, then alternates between long parent
    # edges and new milestone creation until the chain can be connected.
    left_samples, left_file = run_eq_window(
        bin_path, ctx, left_center, float(combo["k_pull"]), int(fixed["eq_steps"]), int(fixed["eq_nout"]),
        next_seed, out_root / "milestone_0",
    )
    milestones.append({
        "milestone_id": 0,
        "side": "left",
        "depth": 0,
        "center_x": left_center,
        "k_eq": float(combo["k_pull"]),
        "eq_steps": int(fixed["eq_steps"]),
        "eq_nout": int(fixed["eq_nout"]),
        "cumulative_start": cumulative,
        "cumulative_end": cumulative + int(fixed["eq_steps"]),
        "seed": next_seed,
        "eq_file": "eq_window.csv",
        "milestone_dir": "milestone_0",
    })
    left_points.append({"id": 0, "center": left_center, "samples": left_samples, "dir": out_root / "milestone_0"})
    cumulative += int(fixed["eq_steps"])
    next_seed += 1

    right_samples, right_file = run_eq_window(
        bin_path, ctx, right_center, float(combo["k_pull"]), int(fixed["eq_steps"]), int(fixed["eq_nout"]),
        next_seed, out_root / "milestone_1",
    )
    milestones.append({
        "milestone_id": 1,
        "side": "right",
        "depth": 0,
        "center_x": right_center,
        "k_eq": float(combo["k_pull"]),
        "eq_steps": int(fixed["eq_steps"]),
        "eq_nout": int(fixed["eq_nout"]),
        "cumulative_start": cumulative,
        "cumulative_end": cumulative + int(fixed["eq_steps"]),
        "seed": next_seed,
        "eq_file": "eq_window.csv",
        "milestone_dir": "milestone_1",
    })
    right_points.append({"id": 1, "center": right_center, "samples": right_samples, "dir": out_root / "milestone_1"})
    cumulative += int(fixed["eq_steps"])
    next_seed += 1

    edges: list[dict] = []
    next_milestone_id = 2
    stop_reason = "max_depth"

    for depth in range(int(fixed["max_depth_per_side"]) + 1):
        left_frontier = left_points[-1]
        right_frontier = right_points[-1]
        eq_overlap = overlap_coefficient(left_frontier["samples"], right_frontier["samples"], grid)
        if eq_overlap >= float(fixed["overlap_min"]):
            stop_reason = "eq_overlap"
            break

        # Parent edges probe whether the current two frontiers already have
        # enough nonequilibrium connectivity to stop growth.
        parent_dir = out_root / f"parent_{depth}"
        run_neq_edge(
            bin_path,
            ctx,
            float(left_frontier["center"]),
            float(right_frontier["center"]),
            left_frontier["dir"] / left_file,
            right_frontier["dir"] / right_file,
            float(combo["k_pull"]),
            int(fixed["n_traj_per_direction"]),
            int(fixed["t_neq"]),
            int(fixed["neq_nout"]),
            next_seed,
            parent_dir,
        )
        parent_cost = 2 * int(fixed["n_traj_per_direction"]) * int(fixed["t_neq"])
        fwd_files = sorted(parent_dir.glob("neq_fwd_*.csv"))
        bwd_files = sorted(parent_dir.glob("neq_bwd_*.csv"))
        left_reach, left_acc = weighted_reach(
            fwd_files, grid, beta, "left", float(left_frontier["center"]), float(fixed["ess_min"])
        )
        right_reach, right_acc = weighted_reach(
            bwd_files, grid, beta, "right", float(right_frontier["center"]), float(fixed["ess_min"])
        )
        work_ov = work_overlap(left_acc["final_works"], right_acc["final_works"])
        cumulative_start = cumulative
        cumulative += parent_cost
        edges.append({
            "edge_id": len(edges),
            "type": "parent",
            "depth": depth,
            "left_id": left_frontier["id"],
            "right_id": right_frontier["id"],
            "left_x": left_frontier["center"],
            "right_x": right_frontier["center"],
            "k_pull": combo["k_pull"],
            "t_neq": fixed["t_neq"],
            "n_traj_per_direction": fixed["n_traj_per_direction"],
            "cumulative_start": cumulative_start,
            "cumulative_end": cumulative,
            "eq_overlap": eq_overlap,
            "left_reach": "" if left_reach is None else left_reach,
            "right_reach": "" if right_reach is None else right_reach,
            "work_overlap": work_ov,
            "resolved": "true" if (work_ov >= float(fixed["work_overlap_min"]) or (left_reach is not None and right_reach is not None and left_reach >= right_reach)) else "false",
            "edge_dir": f"parent_{depth}",
        })
        write_json(parent_dir / "forward_accumulators.json", left_acc)
        write_json(parent_dir / "backward_accumulators.json", right_acc)

        if work_ov >= float(fixed["work_overlap_min"]) or (
            left_reach is not None and right_reach is not None and left_reach >= right_reach
        ):
            stop_reason = "neq_connectivity"
            break

        progressed = False
        if left_reach is not None and left_reach < right_frontier["center"] - 1.0e-9:
            milestone_dir = out_root / f"milestone_{next_milestone_id}"
            samples, _ = run_eq_window(
                bin_path, ctx, left_reach, float(combo["k_pull"]), int(fixed["eq_steps"]), int(fixed["eq_nout"]),
                next_seed, milestone_dir,
            )
            milestones.append({
                "milestone_id": next_milestone_id,
                "side": "left",
                "depth": depth + 1,
                "center_x": left_reach,
                "k_eq": float(combo["k_pull"]),
                "eq_steps": int(fixed["eq_steps"]),
                "eq_nout": int(fixed["eq_nout"]),
                "cumulative_start": cumulative,
                "cumulative_end": cumulative + int(fixed["eq_steps"]),
                "seed": next_seed,
                "eq_file": "eq_window.csv",
                "milestone_dir": f"milestone_{next_milestone_id}",
            })
            left_points.append({"id": next_milestone_id, "center": left_reach, "samples": samples, "dir": milestone_dir})
            cumulative += int(fixed["eq_steps"])
            next_milestone_id += 1
            next_seed += 1
            progressed = True

        if right_reach is not None and right_reach > left_points[-1]["center"] + 1.0e-9:
            milestone_dir = out_root / f"milestone_{next_milestone_id}"
            samples, _ = run_eq_window(
                bin_path, ctx, right_reach, float(combo["k_pull"]), int(fixed["eq_steps"]), int(fixed["eq_nout"]),
                next_seed, milestone_dir,
            )
            milestones.append({
                "milestone_id": next_milestone_id,
                "side": "right",
                "depth": depth + 1,
                "center_x": right_reach,
                "k_eq": float(combo["k_pull"]),
                "eq_steps": int(fixed["eq_steps"]),
                "eq_nout": int(fixed["eq_nout"]),
                "cumulative_start": cumulative,
                "cumulative_end": cumulative + int(fixed["eq_steps"]),
                "seed": next_seed,
                "eq_file": "eq_window.csv",
                "milestone_dir": f"milestone_{next_milestone_id}",
            })
            right_points.append({"id": next_milestone_id, "center": right_reach, "samples": samples, "dir": milestone_dir})
            cumulative += int(fixed["eq_steps"])
            next_milestone_id += 1
            next_seed += 1
            progressed = True

        if not progressed:
            stop_reason = "no_new_milestones"
            break

    # Once growth stops, add nearest-neighbor edges along the discovered chain
    # so the reducer can stitch local segments into an absolute PMF.
    chain = left_points + list(reversed(right_points))
    for idx in range(len(chain) - 1):
        left_m = chain[idx]
        right_m = chain[idx + 1]
        edge_dir = out_root / f"adjacent_{idx}"
        run_neq_edge(
            bin_path,
            ctx,
            float(left_m["center"]),
            float(right_m["center"]),
            left_m["dir"] / "eq_window.csv",
            right_m["dir"] / "eq_window.csv",
            float(combo["k_pull"]),
            int(fixed["n_traj_per_direction"]),
            int(fixed["t_neq"]),
            int(fixed["neq_nout"]),
            next_seed + idx,
            edge_dir,
        )
        cost = 2 * int(fixed["n_traj_per_direction"]) * int(fixed["t_neq"])
        edges.append({
            "edge_id": len(edges),
            "type": "adjacent",
            "depth": idx,
            "left_id": left_m["id"],
            "right_id": right_m["id"],
            "left_x": left_m["center"],
            "right_x": right_m["center"],
            "k_pull": combo["k_pull"],
            "t_neq": fixed["t_neq"],
            "n_traj_per_direction": fixed["n_traj_per_direction"],
            "cumulative_start": cumulative,
            "cumulative_end": cumulative + cost,
            "eq_overlap": overlap_coefficient(left_m["samples"], right_m["samples"], grid),
            "left_reach": "",
            "right_reach": "",
            "work_overlap": "",
            "resolved": "true",
            "edge_dir": f"adjacent_{idx}",
        })
        cumulative += cost

    write_csv(
        out_root / "milestones.csv",
        [
            "milestone_id", "side", "depth", "center_x", "k_eq", "eq_steps", "eq_nout",
            "cumulative_start", "cumulative_end", "seed", "eq_file", "milestone_dir",
        ],
        milestones,
    )
    write_csv(
        out_root / "edges.csv",
        [
            "edge_id", "type", "depth", "left_id", "right_id", "left_x", "right_x",
            "k_pull", "t_neq", "n_traj_per_direction", "cumulative_start",
            "cumulative_end", "eq_overlap", "left_reach", "right_reach",
            "work_overlap", "resolved", "edge_dir",
        ],
        edges,
    )
    write_json(
        out_root / "mines_summary.json",
        {
            "combo_label": combo_label,
            "seed": seed,
            "k_pull": combo["k_pull"],
            "eq_steps": fixed["eq_steps"],
            "t_neq": fixed["t_neq"],
            "grid_dx": fixed.get("grid_dx", ctx["rmse_eval_grid"]["dx"]),
            "n_traj_per_direction": fixed["n_traj_per_direction"],
            "ess_min": fixed["ess_min"],
            "overlap_min": fixed["overlap_min"],
            "work_overlap_min": fixed["work_overlap_min"],
            "max_depth_per_side": fixed["max_depth_per_side"],
            "stop_reason": stop_reason,
            "n_milestones": len(milestones),
            "n_edges": len(edges),
            "total_steps": cumulative,
            "chain_centers": [item["center"] for item in chain],
        },
    )


def run_mines_current_protocol(
    system_root: Path,
    seed: int,
    bin_path: str,
    label: str,
    t_neq_override: int | None = None,
    n_neq_traj_override: int | None = None,
    total_budget_steps_override: int | None = None,
) -> None:
    ctx = load_json(system_root / "run_context.json")
    out_root = system_root / "MINES" / label / "raw" / f"seed_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    params = {
        "k_min": 1.0,
        "k_max": 50.0,
        "k_rescue": 10.0,
        "n_eq_steps": 10000,
        "eq_save_every": 10,
        "n_neq_traj": 100,
        "t_neq": 5000,
        "neq_save_every": 1,
        "tail_fraction": 0.9,
        "q_next": 0.9,
        "alpha": 2.0,
        "x_method_leap": "leap-fixed",
        "x_leap": 1.5,
        "m_max": 10,
        "bin_width": 0.1,
        "x_method_target": "target-next",
        "k_method_leap": "Force-matching",
        "x_method_neq": "linear",
        "k_method_neq": "square-root-linear",
        "js_divergence_stop": 0.8,
        "total_budget_steps": 200000,
    }
    if t_neq_override is not None:
        params["t_neq"] = int(t_neq_override)
    if n_neq_traj_override is not None:
        params["n_neq_traj"] = int(n_neq_traj_override)
    if total_budget_steps_override is not None:
        params["total_budget_steps"] = int(total_budget_steps_override)
    grid = build_grid(
        float(ctx["grid"]["xmin"]),
        float(ctx["grid"]["xmax"]),
        float(params["bin_width"]),
    )
    dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    analytic_pmf = analytic_doublewell_profile_grid(grid, ctx)
    kT = float(ctx["thermal_kT"])
    pymbar_module = load_pymbar()

    windows_root = out_root / "windows"
    generations_root = out_root / "generations"
    windows_root.mkdir(parents=True, exist_ok=True)
    generations_root.mkdir(parents=True, exist_ok=True)

    eq_nout = max(int(params["n_eq_steps"] // params["eq_save_every"]), 1)
    eq_stage_cost = float(int(params["n_eq_steps"]))
    neq_direction_cost = float(int(params["t_neq"]) + 2)
    side_pair_cost = float(2 * (int(params["t_neq"]) + 2))
    generic_pair_cost = float(2 * (int(params["t_neq"]) + 2))
    total_budget_steps = float(params["total_budget_steps"])

    def make_window(
        name: str,
        center_x: float,
        k_value: float,
        start_xy: tuple[float, float] | None,
        eq_seed: int,
    ) -> dict:
        window_root = windows_root / name.lower()
        run_eq_window(
            bin_path,
            ctx,
            float(center_x),
            float(k_value),
            int(params["n_eq_steps"]),
            eq_nout,
            eq_seed,
            window_root,
            start_xy=start_xy,
        )
        eq_rows = read_csv_rows(window_root / "eq_window.csv")
        if not eq_rows:
            raise RuntimeError(f"MiNES current protocol produced no EQ samples for {name}.")
        tail_rows = retained_tail_rows(eq_rows, float(params["tail_fraction"]))
        write_csv(window_root / "eq_tail.csv", list(eq_rows[0].keys()), tail_rows)
        x_most = x_most_from_tail_rows(tail_rows, grid)
        summary = {
            "name": name,
            "center_x": float(center_x),
            "k": float(k_value),
            "start_xy": None if start_xy is None else [float(start_xy[0]), float(start_xy[1])],
            "eq_file": str((window_root / "eq_window.csv").relative_to(out_root)),
            "tail_file": str((window_root / "eq_tail.csv").relative_to(out_root)),
            "n_eq_samples": int(len(eq_rows)),
            "n_tail_samples": int(len(tail_rows)),
            "x_most": float(x_most),
            "base_eq_step_max": float(max(float(row["step"]) for row in eq_rows)) if eq_rows else 0.0,
            "total_eq_step_max": float(max(float(row["step"]) for row in eq_rows)) if eq_rows else 0.0,
            "eq_extension_files": [],
        }
        write_json(window_root / "window_summary.json", summary)
        tail_rows_annotated = []
        for idx, row in enumerate(tail_rows):
            annotated = dict(row)
            annotated["source_tail_index"] = int(idx)
            tail_rows_annotated.append(annotated)
        return {
            "name": name,
            "center_x": float(center_x),
            "k": float(k_value),
            "x_most": float(x_most),
            "eq_rows": eq_rows,
            "tail_rows": tail_rows,
            "tail_rows_annotated": tail_rows_annotated,
            "root": window_root,
            "eq_file": str((window_root / "eq_window.csv").relative_to(out_root)),
            "tail_file": str((window_root / "eq_tail.csv").relative_to(out_root)),
        }

    def extend_window_eq(
        window: dict,
        extension_steps: int,
        round_idx: int,
        eq_seed: int,
    ) -> dict:
        extension_steps = int(extension_steps)
        if extension_steps <= 0:
            return window
        window_root = Path(window["root"])
        extension_root = window_root / f"refine_eq_round_{round_idx:02d}"
        last_eq_row = window["eq_rows"][-1] if window["eq_rows"] else None
        start_xy = (
            (float(last_eq_row["x"]), float(last_eq_row["y"]))
            if last_eq_row is not None
            else (float(window["center_x"]), 0.0)
        )
        extension_nout = max(int(extension_steps // params["eq_save_every"]), 1)
        run_eq_window(
            bin_path,
            ctx,
            float(window["center_x"]),
            float(window["k"]),
            extension_steps,
            extension_nout,
            eq_seed,
            extension_root,
            start_xy=start_xy,
        )
        extension_rows = read_csv_rows(extension_root / "eq_window.csv")
        if not extension_rows:
            return window

        base_last_step = float(window["eq_rows"][-1]["step"]) if window["eq_rows"] else -float(params["eq_save_every"])
        adjusted_extension_rows = []
        for row in extension_rows:
            adjusted = dict(row)
            adjusted["step"] = int(round(base_last_step + float(params["eq_save_every"]) + float(row["step"])))
            adjusted_extension_rows.append(adjusted)

        extension_file = window_root / f"eq_extension_{round_idx:02d}.csv"
        write_csv(extension_file, list(adjusted_extension_rows[0].keys()), adjusted_extension_rows)

        combined_eq_rows = list(window["eq_rows"]) + adjusted_extension_rows
        tail_rows = retained_tail_rows(combined_eq_rows, float(params["tail_fraction"]))
        write_csv(window_root / "eq_tail.csv", list(combined_eq_rows[0].keys()), tail_rows)
        x_most = x_most_from_tail_rows(tail_rows, grid)

        summary = load_json(window_root / "window_summary.json")
        extension_files = list(summary.get("eq_extension_files", []))
        extension_files.append(str(extension_file.relative_to(out_root)))
        summary.update(
            {
                "tail_file": str((window_root / "eq_tail.csv").relative_to(out_root)),
                "n_eq_samples": int(len(combined_eq_rows)),
                "n_tail_samples": int(len(tail_rows)),
                "x_most": float(x_most),
                "total_eq_step_max": float(max(float(row["step"]) for row in combined_eq_rows)),
                "eq_extension_files": extension_files,
            }
        )
        write_json(window_root / "window_summary.json", summary)

        tail_rows_annotated = []
        for idx, row in enumerate(tail_rows):
            annotated = dict(row)
            annotated["source_tail_index"] = int(idx)
            tail_rows_annotated.append(annotated)

        window["eq_rows"] = combined_eq_rows
        window["tail_rows"] = tail_rows
        window["tail_rows_annotated"] = tail_rows_annotated
        window["x_most"] = float(x_most)
        return window

    def load_existing_window(name: str) -> dict:
        window_root = windows_root / name.lower()
        eq_rows = read_csv_rows(window_root / "eq_window.csv")
        tail_rows = read_csv_rows(window_root / "eq_tail.csv")
        summary = load_json(window_root / "window_summary.json")
        tail_rows_annotated = []
        for idx, row in enumerate(tail_rows):
            annotated = dict(row)
            annotated["source_tail_index"] = int(idx)
            tail_rows_annotated.append(annotated)
        return {
            "name": str(summary["name"]),
            "center_x": float(summary["center_x"]),
            "k": float(summary["k"]),
            "x_most": float(summary["x_most"]),
            "eq_rows": eq_rows,
            "tail_rows": tail_rows,
            "tail_rows_annotated": tail_rows_annotated,
            "root": window_root,
            "eq_file": str((window_root / "eq_window.csv").relative_to(out_root)),
            "tail_file": str((window_root / "eq_tail.csv").relative_to(out_root)),
        }

    def load_existing_forward_generation_side(
        generation_idx: int,
        side: str,
        parent_window: dict,
        opposite_window: dict,
    ) -> dict | None:
        side_root = generations_root / f"g{generation_idx:02d}" / side
        base_path_file = side_root / "protocols" / "base_forward_path.csv"
        forward_base_dir = side_root / "forward_base"
        if not base_path_file.exists() or not forward_base_dir.exists():
            return None
        draw_path = forward_base_dir / "drawn_start_samples.csv"
        endpoint_path = forward_base_dir / "forward_endpoints.csv"
        if not draw_path.exists() or not endpoint_path.exists():
            return None
        trajectories: list[list[dict[str, str]]] = []
        traj_paths = sorted(forward_base_dir.glob("traj_*/neq_fwd_0.csv"))
        if not traj_paths:
            return None
        for traj_path in traj_paths:
            traj_rows = read_csv_rows(traj_path)
            if not traj_rows:
                return None
            trajectories.append(traj_rows)
        draw_rows = read_csv_rows(draw_path)
        endpoint_rows = read_csv_rows(endpoint_path)
        final_x = np.asarray([float(row["final_x"]) for row in endpoint_rows], dtype=float)
        proposal = design_force_matched_child(
            side=side,
            final_x=final_x,
            parent_tail_rows=parent_window["tail_rows"],
            trajectories=trajectories,
            path_rows=read_protocol_path_rows(base_path_file),
            parent_center=float(parent_window["center_x"]),
            parent_x_most=float(parent_window["x_most"]),
            opposite_center=float(opposite_window["center_x"]),
            opposite_k=float(opposite_window["k"]),
            alpha=float(params["alpha"]),
            x_method_leap=str(params["x_method_leap"]),
            x_leap=float(params["x_leap"]),
            q_next_level=float(params["q_next"]),
            k_min=float(params["k_min"]),
            k_max=float(params["k_max"]),
            target_rule=str(params["x_method_target"]),
            k_method_leap=str(params["k_method_leap"]),
            grid=grid,
            ctx=ctx,
        )
        return {
            "side": side,
            "side_root": side_root,
            "base_path_file": base_path_file,
            "base_path_rows": read_protocol_path_rows(base_path_file),
            "forward_base_dir": forward_base_dir,
            "draw_rows": draw_rows,
            "endpoint_rows": endpoint_rows,
            "trajectories": trajectories,
            "proposal": proposal,
            "parent_window": parent_window,
            "opposite_window": opposite_window,
        }

    def write_overlap_mbar_summary(
        generation_idx: int,
        left_window: dict,
        right_window: dict,
        overlap_js: float,
        mode: str = "child_overlap_mbar",
        root_name: str = "child_overlap_mbar",
        overlap_root: Path | None = None,
    ) -> dict:
        if overlap_root is None:
            overlap_root = generations_root / f"g{generation_idx:02d}" / root_name
        overlap_root.mkdir(parents=True, exist_ok=True)
        left_values = np.asarray([float(row["x"]) for row in left_window["tail_rows"]], dtype=float)
        right_values = np.asarray([float(row["x"]) for row in right_window["tail_rows"]], dtype=float)
        pmf, _, _, ess = us_mbar_profile_details(
            [left_values, right_values],
            [float(left_window["center_x"]), float(right_window["center_x"])],
            [float(left_window["k"]), float(right_window["k"])],
            grid,
            ctx,
            None,
        )
        left_idx = int(np.argmin(np.abs(grid - float(left_window["x_most"]))))
        right_idx = int(np.argmin(np.abs(grid - float(right_window["x_most"]))))
        anchor_delta_f = float("nan")
        if np.isfinite(pmf[left_idx]) and np.isfinite(pmf[right_idx]):
            anchor_delta_f = float(pmf[right_idx] - pmf[left_idx])
        pmf_path = overlap_root / "overlap_mbar_pmf.csv"
        write_csv(
            pmf_path,
            ["x", "mbar_pmf", "ess", "analytic_pmf"],
            [
                {
                    "x": float(x),
                    "mbar_pmf": "" if not math.isfinite(float(free_energy)) else float(free_energy),
                    "ess": "" if not math.isfinite(float(local_ess)) else float(local_ess),
                    "analytic_pmf": float(analytic),
                }
                for x, free_energy, local_ess, analytic in zip(
                    grid.tolist(),
                    pmf.tolist(),
                    ess.tolist(),
                    analytic_pmf.tolist(),
                )
            ],
        )
        summary = {
            "generation": int(generation_idx),
            "mode": str(mode),
            "left_window": {
                "name": str(left_window["name"]),
                "center_x": float(left_window["center_x"]),
                "k": float(left_window["k"]),
                "x_most": float(left_window["x_most"]),
                "tail_file": left_window["tail_file"],
            },
            "right_window": {
                "name": str(right_window["name"]),
                "center_x": float(right_window["center_x"]),
                "k": float(right_window["k"]),
                "x_most": float(right_window["x_most"]),
                "tail_file": right_window["tail_file"],
            },
            "left_child": {
                "name": str(left_window["name"]),
                "center_x": float(left_window["center_x"]),
                "k": float(left_window["k"]),
                "x_most": float(left_window["x_most"]),
                "tail_file": left_window["tail_file"],
            },
            "right_child": {
                "name": str(right_window["name"]),
                "center_x": float(right_window["center_x"]),
                "k": float(right_window["k"]),
                "x_most": float(right_window["x_most"]),
                "tail_file": right_window["tail_file"],
            },
            "js_divergence": float(overlap_js),
            "threshold": float(params["js_divergence_stop"]),
            "overlap_coefficient": overlap_coefficient(
                left_values.tolist(),
                right_values.tolist(),
                grid,
            ),
            "anchor_delta_f": anchor_delta_f,
            "pmf_file": str(pmf_path.relative_to(out_root)),
        }
        write_json(overlap_root / "overlap_summary.json", summary)
        return summary

    def write_rescue_overlap_mbar_step(
        generation_idx: int,
        rescue_idx: int,
        left_window: dict,
        right_window: dict,
        base_segment_root: Path,
        overlap_js: float,
    ) -> dict:
        rescue_root = generations_root / f"g{generation_idx:02d}" / rescue_dirname(rescue_idx)
        rescue_root.mkdir(parents=True, exist_ok=True)
        overlap_root = rescue_root / "eq_overlap_mbar"
        overlap_summary = write_overlap_mbar_summary(
            generation_idx,
            left_window,
            right_window,
            overlap_js,
            mode="eq_overlap_mbar",
            root_name="eq_overlap_mbar",
            overlap_root=overlap_root,
        )
        rescue_summary = {
            "generation": int(generation_idx),
            "rescue_index": int(rescue_idx),
            "mode": "eq_overlap_mbar",
            "left_window": {
                "name": str(left_window["name"]),
                "center_x": float(left_window["center_x"]),
                "k": float(left_window["k"]),
                "x_most": float(left_window["x_most"]),
                "tail_file": left_window["tail_file"],
            },
            "right_window": {
                "name": str(right_window["name"]),
                "center_x": float(right_window["center_x"]),
                "k": float(right_window["k"]),
                "x_most": float(right_window["x_most"]),
                "tail_file": right_window["tail_file"],
            },
            "base_segment_dir": str(base_segment_root.relative_to(out_root)),
            "js_divergence": float(overlap_js),
            "threshold": float(params["js_divergence_stop"]),
            "overlap_coefficient": float(overlap_summary["overlap_coefficient"]),
            "replacement_segment_mode": "eq_overlap_mbar",
            "overlap_summary_dir": str(overlap_root.relative_to(out_root)),
            "overlap_summary_file": str((overlap_root / "overlap_summary.json").relative_to(out_root)),
        }
        if int(rescue_idx) == 1:
            rescue_summary["initial_base_segment_dir"] = str(base_segment_root.relative_to(out_root))
            rescue_summary["initial_middle_segment_dir"] = str(base_segment_root.relative_to(out_root))
        write_json(rescue_root / "rescue_summary.json", rescue_summary)
        return {
            "summary": rescue_summary,
            "rescue_root": rescue_root,
            "overlap_root": overlap_root,
            "overlap_summary": overlap_summary,
            "base_segment_root": base_segment_root,
        }

    def write_endpoint_overlap_stop_summary(
        generation_idx: int,
        left_state: dict,
        right_state: dict,
        left_overlap: float,
        right_overlap: float,
        average_overlap: float,
    ) -> dict:
        stop_root = generations_root / f"g{generation_idx:02d}" / "endpoint_overlap_stop"
        stop_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "generation": int(generation_idx),
            "mode": "endpoint_overlap_stop",
            "threshold": float(params["endpoint_overlap_stop"]),
            "left_overlap_vs_right_parent": float(left_overlap),
            "right_overlap_vs_left_parent": float(right_overlap),
            "average_overlap": float(average_overlap),
            "left_parent": {
                "name": str(left_state["parent_window"]["name"]),
                "center_x": float(left_state["parent_window"]["center_x"]),
                "k": float(left_state["parent_window"]["k"]),
                "x_most": float(left_state["parent_window"]["x_most"]),
                "tail_file": left_state["parent_window"]["tail_file"],
            },
            "right_parent": {
                "name": str(right_state["parent_window"]["name"]),
                "center_x": float(right_state["parent_window"]["center_x"]),
                "k": float(right_state["parent_window"]["k"]),
                "x_most": float(right_state["parent_window"]["x_most"]),
                "tail_file": right_state["parent_window"]["tail_file"],
            },
            "left_forward": {
                "name": str(left_state["parent_window"]["name"]),
                "opposite_name": str(left_state["opposite_window"]["name"]),
                "endpoints_file": str(
                    (left_state["forward_base_dir"] / "forward_endpoints.csv").relative_to(out_root)
                ),
            },
            "right_forward": {
                "name": str(right_state["parent_window"]["name"]),
                "opposite_name": str(right_state["opposite_window"]["name"]),
                "endpoints_file": str(
                    (right_state["forward_base_dir"] / "forward_endpoints.csv").relative_to(out_root)
                ),
            },
        }
        write_json(stop_root / "overlap_summary.json", summary)
        return summary

    def write_target_cross_connection_summary(
        generation_idx: int,
        left_state: dict,
        right_state: dict,
        left_target: float,
        right_target: float,
        left_connected: bool,
        right_connected: bool,
    ) -> dict:
        stop_root = generations_root / f"g{generation_idx:02d}" / "target_cross_connected"
        stop_root.mkdir(parents=True, exist_ok=True)
        left_parent_x_most = float(left_state["parent_window"]["x_most"])
        right_parent_x_most = float(right_state["parent_window"]["x_most"])
        summary = {
            "generation": int(generation_idx),
            "mode": "target_cross_connected",
            "left_target_x": float(left_target),
            "right_target_x": float(right_target),
            "left_parent": {
                "name": str(left_state["parent_window"]["name"]),
                "center_x": float(left_state["parent_window"]["center_x"]),
                "k": float(left_state["parent_window"]["k"]),
                "x_most": left_parent_x_most,
                "tail_file": left_state["parent_window"]["tail_file"],
            },
            "right_parent": {
                "name": str(right_state["parent_window"]["name"]),
                "center_x": float(right_state["parent_window"]["center_x"]),
                "k": float(right_state["parent_window"]["k"]),
                "x_most": right_parent_x_most,
                "tail_file": right_state["parent_window"]["tail_file"],
            },
            "left_condition": {
                "rule": "x_target_L_next > x_most_R_parent",
                "lhs": float(left_target),
                "rhs": right_parent_x_most,
                "satisfied": bool(left_connected),
            },
            "right_condition": {
                "rule": "x_target_R_next < x_most_L_parent",
                "lhs": float(right_target),
                "rhs": left_parent_x_most,
                "satisfied": bool(right_connected),
            },
        }
        write_json(stop_root / "connection_summary.json", summary)
        return summary

    def write_variance_bridge_summary(
        generation_idx: int,
        left_window: dict,
        right_window: dict,
        left_state: dict,
    ) -> dict:
        bridge_root = generations_root / f"g{generation_idx:02d}" / "variance_bridge"
        bridge_root.mkdir(parents=True, exist_ok=True)

        left_values = np.asarray([float(row["x"]) for row in left_window["tail_rows"]], dtype=float)
        right_values = np.asarray([float(row["x"]) for row in right_window["tail_rows"]], dtype=float)
        if left_values.size == 0 or right_values.size == 0:
            raise RuntimeError("MiNES variance bridge needs both left and right EQ tail samples.")

        left_center = float(left_window["center_x"])
        right_center = float(right_window["center_x"])
        left_k = float(left_window["k"])
        right_k = float(right_window["k"])
        base_pmf, _, _, _ = us_mbar_profile_details(
            [left_values, right_values],
            [left_center, right_center],
            [left_k, right_k],
            grid,
            ctx,
            None,
        )
        base_pmf_ref0, _ = align_pmf_to_value(base_pmf, grid, float(left_window["x_most"]), 0.0)

        n_boot = 32
        rng = np.random.default_rng(seed + 65000 + generation_idx)
        boot_stack: list[np.ndarray] = []
        for _ in range(n_boot):
            boot_left = rng.choice(left_values, size=left_values.size, replace=True)
            boot_right = rng.choice(right_values, size=right_values.size, replace=True)
            boot_pmf, _, _, _ = us_mbar_profile_details(
                [boot_left, boot_right],
                [left_center, right_center],
                [left_k, right_k],
                grid,
                ctx,
                None,
            )
            boot_pmf_ref0, _ = align_pmf_to_value(boot_pmf, grid, float(left_window["x_most"]), 0.0)
            boot_stack.append(boot_pmf_ref0)

        boot_stack_arr = np.vstack(boot_stack)
        with np.errstate(invalid="ignore"):
            boot_var = np.nanvar(boot_stack_arr, axis=0, ddof=1 if boot_stack_arr.shape[0] > 1 else 0)

        anchor_lo = min(float(left_window["x_most"]), float(right_window["x_most"]))
        anchor_hi = max(float(left_window["x_most"]), float(right_window["x_most"]))
        lo = min(left_center, right_center)
        hi = max(left_center, right_center)
        interval_mask = np.isfinite(boot_var) & (grid >= lo) & (grid <= hi)
        if not np.any(interval_mask):
            interval_mask = np.isfinite(boot_var) & (grid >= anchor_lo) & (grid <= anchor_hi)
        if not np.any(interval_mask):
            interval_mask = np.isfinite(boot_var)
        if not np.any(interval_mask):
            raise RuntimeError("MiNES variance bridge could not resolve any finite PMF variance bins.")

        candidate_indices = np.flatnonzero(interval_mask)
        bridge_idx = int(candidate_indices[int(np.argmax(boot_var[candidate_indices]))])
        bridge_variance_peak_x = float(grid[bridge_idx])
        interior_mask = np.isfinite(boot_var) & (grid > left_center) & (grid < right_center)
        interior_indices = np.flatnonzero(interior_mask)
        if interior_indices.size > 0:
            selected_idx = int(interior_indices[int(np.argmin(np.abs(grid[interior_indices] - bridge_variance_peak_x)))])
            bridge_x = float(grid[selected_idx])
        else:
            bridge_x = float(bridge_variance_peak_x)
            if bridge_x <= left_center:
                bridge_x = float(np.nextafter(left_center, right_center))
            elif bridge_x >= right_center:
                bridge_x = float(np.nextafter(right_center, left_center))
        fit_x, fit_y = local_fit_window(grid, base_pmf_ref0, bridge_x, 7)
        fit_geometry = poly_4term_geometry(fit_x, fit_y, bridge_x)
        curvature = float(fit_geometry["curvature"]) if fit_geometry is not None else float("nan")
        if math.isfinite(curvature) and curvature > 0.0:
            bridge_k = float(min(max(abs(curvature) + float(params["k_min"]), float(params["k_min"])), float(params["k_max"])))
            bridge_k_rule = "curvature_plus_k_min"
            bridge_k_clamped_to = "none"
            if abs(curvature) + float(params["k_min"]) < float(params["k_min"]):
                bridge_k_clamped_to = "k_min"
            elif abs(curvature) + float(params["k_min"]) > float(params["k_max"]):
                bridge_k_clamped_to = "k_max"
        else:
            bridge_k = float(params["k_min"])
            bridge_k_rule = "k_min_fallback_nonpositive_curvature"
            bridge_k_clamped_to = "k_min_fallback"

        selection = select_low_work_start_candidate(
            collect_saved_neq_candidates(left_state["trajectories"], "left"),
            bridge_x,
            grid,
        )
        if not selection:
            raise RuntimeError("MiNES variance bridge could not select a low-work EQ start frame.")

        bridge_window = make_window(
            "M",
            bridge_x,
            bridge_k,
            (float(selection["x"]), float(selection["y"])),
            seed + 70000 + generation_idx,
        )
        write_csv(
            bridge_root / "bridge_variance_profile.csv",
            ["x", "pmf_ref0", "boot_var", "analytic_pmf"],
            [
                {
                    "x": float(x),
                    "pmf_ref0": "" if not math.isfinite(float(pmf)) else float(pmf),
                    "boot_var": "" if not math.isfinite(float(var)) else float(var),
                    "analytic_pmf": float(analytic),
                }
                for x, pmf, var, analytic in zip(
                    grid.tolist(),
                    base_pmf_ref0.tolist(),
                    boot_var.tolist(),
                    analytic_pmf.tolist(),
                )
            ],
        )
        summary = {
            "generation": int(generation_idx),
            "mode": "variance_bridge",
            "left_window": {
                "name": str(left_window["name"]),
                "center_x": float(left_center),
                "k": float(left_k),
                "x_most": float(left_window["x_most"]),
                "tail_file": left_window["tail_file"],
            },
            "right_window": {
                "name": str(right_window["name"]),
                "center_x": float(right_center),
                "k": float(right_k),
                "x_most": float(right_window["x_most"]),
                "tail_file": right_window["tail_file"],
            },
            "bridge_window": {
                "name": str(bridge_window["name"]),
                "center_x": float(bridge_window["center_x"]),
                "k": float(bridge_window["k"]),
                "x_most": float(bridge_window["x_most"]),
                "eq_file": bridge_window["eq_file"],
                "tail_file": bridge_window["tail_file"],
            },
            "bridge_x": float(bridge_x),
            "bridge_k": float(bridge_k),
            "bridge_k_rule": str(bridge_k_rule),
            "bridge_k_clamped_to": str(bridge_k_clamped_to),
            "bridge_variance_peak_x": float(bridge_variance_peak_x),
            "bridge_curvature": float(curvature),
            "bridge_fit_points": int(len(fit_x)),
            "bridge_interval_left": float(lo),
            "bridge_interval_right": float(hi),
            "bridge_anchor_left": float(anchor_lo),
            "bridge_anchor_right": float(anchor_hi),
            "bridge_bootstrap_count": int(boot_stack_arr.shape[0]),
            "bridge_selection": {
                "source_side": "left",
                "draw_idx": int(selection["draw_idx"]),
                "row_idx": int(selection["row_idx"]),
                "step": int(selection["step"]),
                "x": float(selection["x"]),
                "y": float(selection["y"]),
                "work": float(selection["work"]),
                "target_x": float(selection["target_x"]),
                "target_bin_x": float(selection["target_bin_x"]),
                "selected_bin_x": float(selection["selected_bin_x"]),
            },
            "variance_profile_file": str((bridge_root / "bridge_variance_profile.csv").relative_to(out_root)),
        }
        write_json(bridge_root / "bridge_summary.json", summary)
        return {
            "summary": summary,
            "window": bridge_window,
            "selection": selection,
            "bridge_root": bridge_root,
        }

    def rescue_dirname(rescue_idx: int) -> str:
        if int(rescue_idx) == 1:
            return "first_rescue"
        if int(rescue_idx) == 2:
            return "second_rescue"
        return f"rescue_{int(rescue_idx):02d}"

    def rescue_name_token(window_name: str) -> str:
        name = str(window_name)
        if name.startswith("M"):
            return name.split("^", 1)[0]
        return name

    def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for idx, flag in enumerate(np.asarray(mask, dtype=bool).tolist()):
            if flag and run_start is None:
                run_start = idx
            elif not flag and run_start is not None:
                runs.append((run_start, idx - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(mask) - 1))
        return runs

    def build_segment_arrays(
        trajectories: list[list[dict[str, str]]],
        n_rows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not trajectories or n_rows <= 0:
            return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
        x = np.asarray(
            [[float(row["x"]) for row in traj[:n_rows]] for traj in trajectories],
            dtype=float,
        )
        work = np.asarray(
            [[float(row["work"]) for row in traj[:n_rows]] for traj in trajectories],
            dtype=float,
        )
        return x, work

    def analyze_rescue_candidate(
        generation_idx: int,
        rescue_idx: int,
        left_window: dict,
        right_window: dict,
        segment_root: Path,
    ) -> dict:
        forward_path_file, reverse_path_file = segment_protocol_path_files(segment_root)
        forward_path_rows = read_protocol_path_rows(forward_path_file)
        reverse_path_rows = read_protocol_path_rows(reverse_path_file)
        forward_trajectories = [
            read_csv_rows(path)
            for path in sorted((segment_root / "forward").glob("traj_*/neq_fwd_0.csv"))
        ]
        reverse_trajectories = [
            read_csv_rows(path)
            for path in sorted((segment_root / "reverse").glob("traj_*/neq_fwd_0.csv"))
        ]
        if not forward_trajectories or not reverse_trajectories:
            raise RuntimeError(
                f"MiNES rescue step {rescue_idx} needs saved bidirectional trajectories in {segment_root}."
            )

        n_time = min(
            len(forward_path_rows),
            len(reverse_path_rows),
            min(len(traj) for traj in forward_trajectories),
            min(len(traj) for traj in reverse_trajectories),
        )
        forward_centers = np.asarray([float(row["center_x"]) for row in forward_path_rows[:n_time]], dtype=float)
        forward_ks = np.asarray([float(row["k"]) for row in forward_path_rows[:n_time]], dtype=float)
        forward_x, forward_work_paths = build_segment_arrays(forward_trajectories, n_time)
        reverse_x, reverse_work_paths = build_segment_arrays(reverse_trajectories, n_time)
        rescue_boot = bootstrap_bidirectional_mts_pmf(
            forward_x[:, :n_time],
            forward_work_paths[:, :n_time],
            reverse_x[:, :n_time],
            reverse_work_paths[:, :n_time],
            forward_centers[:n_time],
            forward_ks[:n_time],
            grid,
            reference_x=float(left_window["x_most"]),
            kT=kT,
            n_boot=32,
            fk_boot=8,
            rng_seed=seed + 80000 + 5000 * int(rescue_idx) + generation_idx,
        )
        base_pmf_ref0 = np.asarray(rescue_boot["pmf_ref0"], dtype=float)
        boot_stack = np.asarray(rescue_boot["boot_pmf_stack"], dtype=float)
        var_ref0 = np.asarray(rescue_boot["var_ref0"], dtype=float)
        if boot_stack.ndim == 2 and boot_stack.shape[0] > 0:
            left_window_zero_stack = []
            right_window_zero_stack = []
            endpoint_mean_zero_stack = []
            for boot_row in boot_stack:
                boot_row = np.asarray(boot_row, dtype=float)
                left_window_zero_stack.append(
                    align_pmf_to_reference(
                        boot_row,
                        grid,
                        float(left_window["x_most"]),
                        reference_value=0.0,
                    )
                )
                right_window_zero_stack.append(
                    align_pmf_to_reference(
                        boot_row,
                        grid,
                        float(right_window["x_most"]),
                        reference_value=0.0,
                    )
                )
                aligned_row = align_pmf_to_endpoint_average_zero(
                    boot_row,
                    grid,
                    float(left_window["x_most"]),
                    float(right_window["x_most"]),
                )
                endpoint_mean_zero_stack.append(aligned_row)
            left_window_zero_stack = np.vstack(left_window_zero_stack)
            right_window_zero_stack = np.vstack(right_window_zero_stack)
            endpoint_zero_stack = np.vstack(endpoint_mean_zero_stack)
            with np.errstate(invalid="ignore"):
                var_left_window_zero = np.nanvar(
                    left_window_zero_stack,
                    axis=0,
                    ddof=1 if left_window_zero_stack.shape[0] > 1 else 0,
                )
                var_right_window_zero = np.nanvar(
                    right_window_zero_stack,
                    axis=0,
                    ddof=1 if right_window_zero_stack.shape[0] > 1 else 0,
                )
                var_min_directional = np.fmin(var_left_window_zero, var_right_window_zero)
                var_endpoint_mean_zero = np.nanvar(
                    endpoint_zero_stack,
                    axis=0,
                    ddof=1 if endpoint_zero_stack.shape[0] > 1 else 0,
                )
        else:
            var_left_window_zero = np.full(len(grid), np.nan, dtype=float)
            var_right_window_zero = np.full(len(grid), np.nan, dtype=float)
            var_min_directional = np.full(len(grid), np.nan, dtype=float)
            var_endpoint_mean_zero = np.full(len(grid), np.nan, dtype=float)

        anchor_lo = min(float(left_window["x_most"]), float(right_window["x_most"]))
        anchor_hi = max(float(left_window["x_most"]), float(right_window["x_most"]))
        interval_indices = np.flatnonzero((grid >= anchor_lo - 1.0e-9) & (grid <= anchor_hi + 1.0e-9))
        if interval_indices.size == 0:
            interval_indices = np.arange(len(grid), dtype=int)
        uncovered_runs = contiguous_runs(~np.isfinite(base_pmf_ref0[interval_indices]))
        target_rule_details: dict = {}
        uncovered_run_length = 0
        if uncovered_runs:
            best_run = max(uncovered_runs, key=lambda item: (item[1] - item[0] + 1, -item[0]))
            midpoint_local = int(round(0.5 * float(best_run[0] + best_run[1])))
            midpoint_local = min(max(midpoint_local, best_run[0]), best_run[1])
            target_idx = int(interval_indices[midpoint_local])
            target_rule = "uncovered_midpoint"
            uncovered_run_length = int(best_run[1] - best_run[0] + 1)
            target_rule_details = {
                "run_local_start": int(best_run[0]),
                "run_local_stop": int(best_run[1]),
                "run_global_start": int(interval_indices[best_run[0]]),
                "run_global_stop": int(interval_indices[best_run[1]]),
            }
        else:
            finite_var_indices = interval_indices[np.isfinite(var_min_directional[interval_indices])]
            if finite_var_indices.size == 0:
                finite_var_indices = np.flatnonzero(np.isfinite(var_min_directional))
            if finite_var_indices.size == 0:
                raise RuntimeError(f"MiNES rescue step {rescue_idx} could not find any finite variance bins.")
            target_idx = int(
                finite_var_indices[int(np.nanargmax(var_min_directional[finite_var_indices]))]
            )
            target_rule = "high_variance_peak"
        target_x = float(grid[target_idx])

        fit_mask = (grid >= anchor_lo - 1.0e-9) & (grid <= anchor_hi + 1.0e-9) & np.isfinite(base_pmf_ref0)
        fit_x = np.asarray(grid[fit_mask], dtype=float)
        fit_y = np.asarray(base_pmf_ref0[fit_mask], dtype=float)
        fit_mode = "resolved_interval_cubic"
        if fit_x.size < 4 or np.unique(fit_x).size < 4:
            fit_x, fit_y = local_fit_window(grid, base_pmf_ref0, target_x, 7)
            fit_mode = "local_cubic_fallback"
        if fit_x.size < 4 or np.unique(fit_x).size < 4:
            raise RuntimeError(f"MiNES rescue step {rescue_idx} could not fit a cubic PMF around the selected target.")
        fit_local_x = np.asarray(fit_x, dtype=float) - target_x
        fit_coeffs = np.polyfit(fit_local_x, np.asarray(fit_y, dtype=float), deg=3)
        fit_slope = float(fit_coeffs[2]) if math.isfinite(float(fit_coeffs[2])) else 0.0
        fit_curvature = float(2.0 * fit_coeffs[1]) if math.isfinite(float(fit_coeffs[1])) else 0.0

        center_lo = min(float(left_window["center_x"]), float(right_window["center_x"]))
        center_hi = max(float(left_window["center_x"]), float(right_window["center_x"]))
        if fit_curvature > 0.0:
            if fit_slope > 0.0:
                rescue_center = float(target_x + float(params["x_leap"]))
                rescue_center_rule = "target_plus_x_leap_positive_slope"
            else:
                rescue_center = float(target_x - float(params["x_leap"]))
                rescue_center_rule = "target_minus_x_leap_nonpositive_slope"
            if rescue_center <= center_lo:
                rescue_center = float(np.nextafter(center_lo, center_hi))
            elif rescue_center >= center_hi:
                rescue_center = float(np.nextafter(center_hi, center_lo))
            signed_gap = float(rescue_center - target_x)
            raw_k = float(fit_slope / signed_gap) if abs(signed_gap) > 1.0e-12 else float("nan")
            if not math.isfinite(raw_k) or raw_k <= 0.0:
                raw_k = float(abs(fit_slope) / max(abs(signed_gap), max(dx, 1.0e-3)))
            rescue_k_rule = "signed_slope_over_signed_gap"
        else:
            rescue_center = float(target_x)
            if rescue_center <= center_lo:
                rescue_center = float(np.nextafter(center_lo, center_hi))
            elif rescue_center >= center_hi:
                rescue_center = float(np.nextafter(center_hi, center_lo))
            if math.isfinite(fit_curvature) and fit_curvature < 0.0:
                raw_k = float((-fit_curvature) + float(params["k_rescue"]))
                rescue_k_rule = "negative_curvature_reflected_plus_k_rescue"
            else:
                raw_k = float(params["k_rescue"])
                rescue_k_rule = "target_center_k_rescue_fallback"
            rescue_center_rule = "center_on_target"
        rescue_k = float(min(max(raw_k, float(params["k_min"])), float(params["k_max"])))
        rescue_k_clamped_to = "none"
        if raw_k < float(params["k_min"]):
            rescue_k_clamped_to = "k_min"
        elif raw_k > float(params["k_max"]):
            rescue_k_clamped_to = "k_max"

        target_var_ref0 = float(var_ref0[target_idx]) if np.isfinite(var_ref0[target_idx]) else float("nan")
        target_var_left_window_zero = (
            float(var_left_window_zero[target_idx])
            if np.isfinite(var_left_window_zero[target_idx])
            else float("nan")
        )
        target_var_right_window_zero = (
            float(var_right_window_zero[target_idx])
            if np.isfinite(var_right_window_zero[target_idx])
            else float("nan")
        )
        target_var_min_directional = (
            float(var_min_directional[target_idx])
            if np.isfinite(var_min_directional[target_idx])
            else float("nan")
        )
        target_var_endpoint_mean_zero = (
            float(var_endpoint_mean_zero[target_idx])
            if np.isfinite(var_endpoint_mean_zero[target_idx])
            else float("nan")
        )

        return {
            "generation": int(generation_idx),
            "rescue_index": int(rescue_idx),
            "left_window": left_window,
            "right_window": right_window,
            "segment_root": segment_root,
            "segment_dir": str(segment_root.relative_to(out_root)),
            "base_pmf_ref0": base_pmf_ref0,
            "var_ref0": var_ref0,
            "var_left_window_zero": var_left_window_zero,
            "var_right_window_zero": var_right_window_zero,
            "var_min_directional": var_min_directional,
            "var_endpoint_mean_zero": var_endpoint_mean_zero,
            "variance_alignment": "min_bidirectional_left_window_zero_right_window_zero",
            "target_idx": int(target_idx),
            "target_x": float(target_x),
            "target_bin_x": float(grid[target_idx]),
            "target_rule": str(target_rule),
            "target_rule_details": dict(target_rule_details),
            "target_var_ref0": float(target_var_ref0),
            "target_var_left_window_zero": float(target_var_left_window_zero),
            "target_var_right_window_zero": float(target_var_right_window_zero),
            "target_var_min_directional": float(target_var_min_directional),
            "target_var_endpoint_mean_zero": float(target_var_endpoint_mean_zero),
            "uncovered_run_length": int(uncovered_run_length),
            "fit_mode": str(fit_mode),
            "fit_coeffs": np.asarray(fit_coeffs, dtype=float).copy(),
            "fit_slope": float(fit_slope),
            "fit_curvature": float(fit_curvature),
            "rescue_center_x": float(rescue_center),
            "rescue_center_rule": str(rescue_center_rule),
            "rescue_k_raw": float(raw_k),
            "rescue_k": float(rescue_k),
            "rescue_k_rule": str(rescue_k_rule),
            "rescue_k_clamped_to": str(rescue_k_clamped_to),
        }

    def rescue_candidate_priority_key(candidate: dict) -> tuple[float, float, float]:
        target_var_min_directional = float(candidate["target_var_min_directional"])
        if not math.isfinite(target_var_min_directional):
            target_var_min_directional = float("-inf")
        uncovered_priority = 1.0 if int(candidate["uncovered_run_length"]) > 0 else 0.0
        return (
            uncovered_priority,
            (
                float(candidate["uncovered_run_length"])
                if uncovered_priority > 0.0
                else target_var_min_directional
            ),
            target_var_min_directional,
        )

    def write_rescue_step(
        generation_idx: int,
        rescue_idx: int,
        left_window: dict,
        right_window: dict,
        base_segment_root: Path,
        overlap_js: float,
        candidate: dict | None = None,
    ) -> dict:
        rescue_root = generations_root / f"g{generation_idx:02d}" / rescue_dirname(rescue_idx)
        rescue_root.mkdir(parents=True, exist_ok=True)
        candidate = candidate or analyze_rescue_candidate(
            generation_idx,
            rescue_idx,
            left_window,
            right_window,
            base_segment_root,
        )

        rescue_eq_start_xy = (float(candidate["rescue_center_x"]), 0.0)
        rescue_window_name = (
            f"M{int(rescue_idx)}^[{rescue_name_token(left_window['name'])}]_"
            f"{{{rescue_name_token(right_window['name'])}}}"
        )
        rescue_window = make_window(
            rescue_window_name,
            float(candidate["rescue_center_x"]),
            float(candidate["rescue_k"]),
            rescue_eq_start_xy,
            seed + 75000 + 5000 * int(rescue_idx) + generation_idx,
        )

        write_csv(
            rescue_root / "rescue_profile.csv",
            [
                "x",
                "pmf_ref0",
                "var_ref0",
                "var_left_window_zero",
                "var_right_window_zero",
                "var_min_directional",
                "var_endpoint_mean_zero",
                "analytic_pmf",
            ],
            [
                {
                    "x": float(x),
                    "pmf_ref0": "" if not math.isfinite(float(pmf_value)) else float(pmf_value),
                    "var_ref0": "" if not math.isfinite(float(ref0_value)) else float(ref0_value),
                    "var_left_window_zero": "" if not math.isfinite(float(left_value)) else float(left_value),
                    "var_right_window_zero": "" if not math.isfinite(float(right_value)) else float(right_value),
                    "var_min_directional": "" if not math.isfinite(float(min_value)) else float(min_value),
                    "var_endpoint_mean_zero": "" if not math.isfinite(float(centered_value)) else float(centered_value),
                    "analytic_pmf": float(analytic),
                }
                for x, pmf_value, ref0_value, left_value, right_value, min_value, centered_value, analytic in zip(
                    grid.tolist(),
                    np.asarray(candidate["base_pmf_ref0"], dtype=float).tolist(),
                    np.asarray(candidate["var_ref0"], dtype=float).tolist(),
                    np.asarray(candidate["var_left_window_zero"], dtype=float).tolist(),
                    np.asarray(candidate["var_right_window_zero"], dtype=float).tolist(),
                    np.asarray(candidate["var_min_directional"], dtype=float).tolist(),
                    np.asarray(candidate["var_endpoint_mean_zero"], dtype=float).tolist(),
                    analytic_pmf.tolist(),
                )
            ],
        )

        left_rescue_root = rescue_root / "left_pair"
        right_rescue_root = rescue_root / "right_pair"
        left_rescue_summary = run_mines_bidirectional_segment_generic(
            system_root=system_root,
            ctx=ctx,
            seed=seed,
            bin_path=bin_path,
            out_root=left_rescue_root,
            left_name=str(left_window["name"]),
            right_name=str(rescue_window["name"]),
            left_center=float(left_window["center_x"]),
            right_center=float(rescue_window["center_x"]),
            left_k=float(left_window["k"]),
            right_k=float(rescue_window["k"]),
            left_tail_rows=left_window["tail_rows"],
            right_tail_rows=rescue_window["tail_rows"],
            left_tail_ref=str(left_window["tail_file"]),
            right_tail_ref=str(rescue_window["tail_file"]),
            t_nes=int(params["t_neq"]),
            n_nes=int(params["n_neq_traj"]),
            use_tail=float(params["tail_fraction"]),
            seed_offset=76000 + 5000 * int(rescue_idx) + 10000 * generation_idx,
        )
        right_rescue_summary = run_mines_bidirectional_segment_generic(
            system_root=system_root,
            ctx=ctx,
            seed=seed,
            bin_path=bin_path,
            out_root=right_rescue_root,
            left_name=str(rescue_window["name"]),
            right_name=str(right_window["name"]),
            left_center=float(rescue_window["center_x"]),
            right_center=float(right_window["center_x"]),
            left_k=float(rescue_window["k"]),
            right_k=float(right_window["k"]),
            left_tail_rows=rescue_window["tail_rows"],
            right_tail_rows=right_window["tail_rows"],
            left_tail_ref=str(rescue_window["tail_file"]),
            right_tail_ref=str(right_window["tail_file"]),
            t_nes=int(params["t_neq"]),
            n_nes=int(params["n_neq_traj"]),
            use_tail=float(params["tail_fraction"]),
            seed_offset=78000 + 5000 * int(rescue_idx) + 10000 * generation_idx,
        )

        rescue_summary = {
            "generation": int(generation_idx),
            "rescue_index": int(rescue_idx),
            "mode": "rescue_window",
            "left_window": {
                "name": str(left_window["name"]),
                "center_x": float(left_window["center_x"]),
                "k": float(left_window["k"]),
                "x_most": float(left_window["x_most"]),
                "tail_file": left_window["tail_file"],
            },
            "right_window": {
                "name": str(right_window["name"]),
                "center_x": float(right_window["center_x"]),
                "k": float(right_window["k"]),
                "x_most": float(right_window["x_most"]),
                "tail_file": right_window["tail_file"],
            },
            "rescue_window": {
                "name": str(rescue_window["name"]),
                "center_x": float(rescue_window["center_x"]),
                "k": float(rescue_window["k"]),
                "x_most": float(rescue_window["x_most"]),
                "eq_file": rescue_window["eq_file"],
                "tail_file": rescue_window["tail_file"],
            },
            "base_segment_dir": str(base_segment_root.relative_to(out_root)),
            "js_divergence": float(overlap_js),
            "threshold": float(params["js_divergence_stop"]),
            "overlap_coefficient": overlap_coefficient(
                [float(row["x"]) for row in left_window["tail_rows"]],
                [float(row["x"]) for row in right_window["tail_rows"]],
                grid,
            ),
            "target_x": float(candidate["target_x"]),
            "target_bin_x": float(candidate["target_bin_x"]),
            "target_rule": str(candidate["target_rule"]),
            "target_rule_details": dict(candidate["target_rule_details"]),
            "variance_alignment": str(candidate["variance_alignment"]),
            "target_var_ref0": float(candidate["target_var_ref0"]),
            "target_var_left_window_zero": float(candidate["target_var_left_window_zero"]),
            "target_var_right_window_zero": float(candidate["target_var_right_window_zero"]),
            "target_var_min_directional": float(candidate["target_var_min_directional"]),
            "target_var_endpoint_mean_zero": float(candidate["target_var_endpoint_mean_zero"]),
            "uncovered_run_length": int(candidate["uncovered_run_length"]),
            "eq_start_rule": "designed_window_center",
            "eq_start_xy": [float(rescue_eq_start_xy[0]), float(rescue_eq_start_xy[1])],
            "fit_mode": str(candidate["fit_mode"]),
            "fit_coeffs_local_cubic": [
                float(value) for value in np.asarray(candidate["fit_coeffs"], dtype=float).tolist()
            ],
            "fit_slope_at_target": float(candidate["fit_slope"]),
            "fit_curvature_at_target": float(candidate["fit_curvature"]),
            "rescue_center_x": float(candidate["rescue_center_x"]),
            "rescue_center_rule": str(candidate["rescue_center_rule"]),
            "rescue_k_raw": float(candidate["rescue_k_raw"]),
            "rescue_k": float(candidate["rescue_k"]),
            "rescue_k_rule": str(candidate["rescue_k_rule"]),
            "rescue_k_clamped_to": str(candidate["rescue_k_clamped_to"]),
            "replacement_segment_mode": "full_bidirectional_segments",
            "rescue_profile_file": str((rescue_root / "rescue_profile.csv").relative_to(out_root)),
            "left_rescue_segment_dir": str(left_rescue_root.relative_to(out_root)),
            "right_rescue_segment_dir": str(right_rescue_root.relative_to(out_root)),
        }
        if int(rescue_idx) == 1:
            rescue_summary["initial_base_segment_dir"] = str(base_segment_root.relative_to(out_root))
            rescue_summary["initial_middle_segment_dir"] = str(base_segment_root.relative_to(out_root))

        write_json(rescue_root / "rescue_summary.json", rescue_summary)
        return {
            "summary": rescue_summary,
            "window": rescue_window,
            "left_segment_summary": left_rescue_summary,
            "right_segment_summary": right_rescue_summary,
            "left_segment_root": left_rescue_root,
            "right_segment_root": right_rescue_root,
            "rescue_root": rescue_root,
            "candidate": candidate,
            "base_segment_root": base_segment_root,
        }

    def run_forward_generation_side(
        generation_idx: int,
        side: str,
        parent_window: dict,
        opposite_window: dict,
        seed_offset: int,
    ) -> dict:
        side_root = generations_root / f"g{generation_idx:02d}" / side
        side_root.mkdir(parents=True, exist_ok=True)
        protocol_dir = side_root / "protocols"
        protocol_dir.mkdir(parents=True, exist_ok=True)
        base_path_file = protocol_dir / "base_forward_path.csv"
        base_path_rows = write_path_linear_sqrtk(
            base_path_file,
            float(parent_window["center_x"]),
            float(opposite_window["center_x"]),
            float(parent_window["k"]),
            float(opposite_window["k"]),
            int(params["t_neq"]),
        )
        start_rows = sample_tail_rows(
            parent_window["tail_rows_annotated"],
            int(params["n_neq_traj"]),
            seed + seed_offset,
        )
        if not start_rows:
            raise RuntimeError(f"MiNES current protocol could not draw any EQ tail starts for {parent_window['name']}.")
        forward_base_dir = side_root / "forward_base"
        draw_rows, endpoint_rows, trajectories = run_single_start_paths(
            bin_path,
            ctx,
            base_path_file,
            start_rows,
            (float(opposite_window["center_x"]), 0.0),
            seed + seed_offset + 1000,
            forward_base_dir,
        )
        write_csv(
            forward_base_dir / "drawn_start_samples.csv",
            ["draw_idx", "source_tail_index", "source_step", "start_x", "start_y", "base_u", "bias_u"],
            draw_rows,
        )
        write_csv(
            forward_base_dir / "forward_endpoints.csv",
            ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
            endpoint_rows,
        )
        final_x = np.asarray([float(row["final_x"]) for row in endpoint_rows], dtype=float)
        proposal = design_force_matched_child(
            side=side,
            final_x=final_x,
            parent_tail_rows=parent_window["tail_rows"],
            trajectories=trajectories,
            path_rows=base_path_rows,
            parent_center=float(parent_window["center_x"]),
            parent_x_most=float(parent_window["x_most"]),
            opposite_center=float(opposite_window["center_x"]),
            opposite_k=float(opposite_window["k"]),
            alpha=float(params["alpha"]),
            x_method_leap=str(params["x_method_leap"]),
            x_leap=float(params["x_leap"]),
            q_next_level=float(params["q_next"]),
            k_min=float(params["k_min"]),
            k_max=float(params["k_max"]),
            target_rule=str(params["x_method_target"]),
            k_method_leap=str(params["k_method_leap"]),
            grid=grid,
            ctx=ctx,
        )
        return {
            "side": side,
            "side_root": side_root,
            "base_path_file": base_path_file,
            "base_path_rows": base_path_rows,
            "forward_base_dir": forward_base_dir,
            "draw_rows": draw_rows,
            "endpoint_rows": endpoint_rows,
            "trajectories": trajectories,
            "proposal": proposal,
            "parent_window": parent_window,
            "opposite_window": opposite_window,
        }

    def finalize_generation_side(
        generation_idx: int,
        side_state: dict,
        child_window: dict,
        selection: dict,
        reverse_seed_offset: int,
    ) -> dict:
        side = str(side_state["side"])
        side_root = Path(side_state["side_root"])
        protocol_dir = side_root / "protocols"
        forward_path_file = protocol_dir / "forward_augmented_path.csv"
        reverse_path_file = protocol_dir / "reverse_path.csv"
        forward_centers = [float(row["center_x"]) for row in side_state["base_path_rows"]] + [float(child_window["center_x"])]
        forward_ks = [float(row["k"]) for row in side_state["base_path_rows"]] + [float(child_window["k"])]
        reverse_centers = [float(child_window["center_x"])] + [float(row["center_x"]) for row in reversed(side_state["base_path_rows"])]
        reverse_ks = [float(child_window["k"])] + [float(row["k"]) for row in reversed(side_state["base_path_rows"])]
        write_protocol_path(forward_path_file, forward_centers, forward_ks)
        write_protocol_path(reverse_path_file, reverse_centers, reverse_ks)
        forward_path_rows = read_protocol_path_rows(forward_path_file)
        reverse_path_rows = read_protocol_path_rows(reverse_path_file)

        synthetic_trajectories, appended_endpoint_rows = append_terminal_bias_rows(
            side_state["trajectories"],
            float(side_state["opposite_window"]["center_x"]),
            float(side_state["opposite_window"]["k"]),
            float(child_window["center_x"]),
            float(child_window["k"]),
        )
        forward_dir = side_root / "forward"
        forward_dir.mkdir(parents=True, exist_ok=True)
        forward_endpoint_rows: list[dict] = []
        for draw_idx, (traj_rows, endpoint_row, draw_row) in enumerate(
            zip(synthetic_trajectories, appended_endpoint_rows, side_state["draw_rows"])
        ):
            traj_dir = forward_dir / f"traj_{draw_idx:03d}"
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_path = traj_dir / "neq_fwd_0.csv"
            write_csv(
                traj_path,
                ["step", "lambda", "x", "y", "base_u", "bias_u", "work"],
                traj_rows,
            )
            forward_endpoint_rows.append(
                {
                    "draw_idx": int(draw_idx),
                    "start_x": float(draw_row["start_x"]),
                    "start_y": float(draw_row["start_y"]),
                    "final_step": int(endpoint_row["final_step"]),
                    "final_lambda": float(endpoint_row["final_lambda"]),
                    "final_x": float(endpoint_row["final_x"]),
                    "final_y": float(endpoint_row["final_y"]),
                    "appended_work": float(endpoint_row["appended_work"]),
                    "terminal_bias_old": float(endpoint_row["terminal_bias_old"]),
                    "terminal_bias_new": float(endpoint_row["terminal_bias_new"]),
                    "traj_file": str(traj_path.relative_to(side_root)),
                }
            )
        write_csv(
            forward_dir / "drawn_start_samples.csv",
            ["draw_idx", "source_tail_index", "source_step", "start_x", "start_y", "base_u", "bias_u"],
            side_state["draw_rows"],
        )
        write_csv(
            forward_dir / "forward_endpoints.csv",
            ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "appended_work", "terminal_bias_old", "terminal_bias_new", "traj_file"],
            forward_endpoint_rows,
        )

        reverse_start_rows = sample_tail_rows(
            child_window["tail_rows_annotated"],
            int(params["n_neq_traj"]),
            seed + reverse_seed_offset,
        )
        reverse_dir = side_root / "reverse"
        reverse_draw_rows, reverse_endpoint_rows, reverse_trajectories = run_single_start_paths(
            bin_path,
            ctx,
            reverse_path_file,
            reverse_start_rows,
            (float(side_state["parent_window"]["center_x"]), 0.0),
            seed + reverse_seed_offset + 1000,
            reverse_dir,
        )
        write_csv(
            reverse_dir / "drawn_start_samples.csv",
            ["draw_idx", "source_tail_index", "source_step", "start_x", "start_y", "base_u", "bias_u"],
            reverse_draw_rows,
        )
        write_csv(
            reverse_dir / "reverse_endpoints.csv",
            ["draw_idx", "start_x", "start_y", "final_step", "final_lambda", "final_x", "final_y", "final_work", "traj_file"],
            reverse_endpoint_rows,
        )

        forward_works = np.asarray([float(row["appended_work"]) for row in forward_endpoint_rows], dtype=float)
        reverse_works = np.asarray([float(row["final_work"]) for row in reverse_endpoint_rows], dtype=float)
        crooks = pymbar_module.other_estimators.bar(forward_works, reverse_works, compute_uncertainty=True)
        if isinstance(crooks, dict):
            delta_f = float(crooks.get("Delta_f", crooks.get("delta_f", float("nan"))))
            delta_f_unc = float(crooks.get("dDelta_f", crooks.get("delta_f_uncertainty", float("nan"))))
        elif isinstance(crooks, tuple):
            delta_f = float(crooks[0])
            delta_f_unc = float(crooks[1]) if len(crooks) > 1 else float("nan")
        else:
            delta_f = float(crooks)
            delta_f_unc = float("nan")

        forward_hs_pmf = hs_reconstruct_oneway_path_rows(synthetic_trajectories, forward_path_rows, grid, ctx)
        reverse_hs_pmf = hs_reconstruct_oneway_path_rows(reverse_trajectories, reverse_path_rows, grid, ctx)
        density_average_pmf = average_pmfs_by_density([forward_hs_pmf, reverse_hs_pmf], dx, kT)
        finite_union = np.isfinite(forward_hs_pmf) | np.isfinite(reverse_hs_pmf)
        density_interval = np.full(len(density_average_pmf), np.nan, dtype=float)
        density_interval[finite_union] = density_average_pmf[finite_union]

        write_csv(
            side_root / "reversible_pmf.csv",
            ["x", "forward_hs_pmf", "reverse_hs_pmf", "density_average_pmf", "analytic_pmf"],
            [
                {
                    "x": float(x),
                    "forward_hs_pmf": "" if not math.isfinite(float(fwd)) else float(fwd),
                    "reverse_hs_pmf": "" if not math.isfinite(float(rev)) else float(rev),
                    "density_average_pmf": "" if not math.isfinite(float(avg)) else float(avg),
                    "analytic_pmf": float(analytic),
                }
                for x, fwd, rev, avg, analytic in zip(
                    grid.tolist(),
                    forward_hs_pmf.tolist(),
                    reverse_hs_pmf.tolist(),
                    density_interval.tolist(),
                    analytic_pmf.tolist(),
                )
            ],
        )

        side_summary = {
            "generation": int(generation_idx),
            "side": side,
            "protocol_variant": str(side_state.get("protocol_variant", "same_side_child")),
            "parent_window": {
                "name": side_state["parent_window"]["name"],
                "center_x": float(side_state["parent_window"]["center_x"]),
                "k": float(side_state["parent_window"]["k"]),
                "x_most": float(side_state["parent_window"]["x_most"]),
                "eq_file": side_state["parent_window"]["eq_file"],
                "tail_file": side_state["parent_window"]["tail_file"],
            },
            "opposite_window": {
                "name": side_state["opposite_window"]["name"],
                "center_x": float(side_state["opposite_window"]["center_x"]),
                "k": float(side_state["opposite_window"]["k"]),
                "x_most": float(side_state["opposite_window"]["x_most"]),
                "eq_file": side_state["opposite_window"]["eq_file"],
                "tail_file": side_state["opposite_window"]["tail_file"],
            },
            "child_window": {
                "name": child_window["name"],
                "center_x": float(child_window["center_x"]),
                "k": float(child_window["k"]),
                "x_most": float(child_window["x_most"]),
                "eq_file": child_window["eq_file"],
                "tail_file": child_window["tail_file"],
                "initialized_from": {
                    "selection_mode": str(selection["selection_mode"]),
                    "source_side": str(selection.get("source_side", side)),
                    "draw_idx": int(selection["draw_idx"]),
                    "row_idx": int(selection["row_idx"]),
                    "step": int(selection["step"]),
                    "x": float(selection["x"]),
                    "y": float(selection["y"]),
                    "work": float(selection["work"]),
                    "target_x": float(selection["target_x"]),
                    "target_bin_x": float(selection["target_bin_x"]),
                    "selected_bin_x": float(selection["selected_bin_x"]),
                },
            },
            "forward_design": {
                "target_x": float(side_state["proposal"]["target_x"]),
                "anchor_x": float(side_state["proposal"]["anchor_x"]),
                "q50_x": float(side_state["proposal"]["q50_x"]),
                "anchor_level": float(side_state["proposal"]["anchor_level"]),
                "target_source": str(side_state["proposal"]["target_source"]),
                "barrier_crossing": bool(side_state["proposal"]["barrier_crossing"]),
                "progress_favorable": bool(side_state["proposal"]["progress_favorable"]),
                "center_raw": float(side_state["proposal"]["center_raw"]),
                "matched_force": float(side_state["proposal"]["matched_force"]),
                "gap": float(side_state["proposal"]["gap"]),
                "raw_k": float(side_state["proposal"]["raw_k"]),
                "k_clamped_to": str(side_state["proposal"]["k_clamped_to"]),
                "design_mode": str(side_state["proposal"]["design_mode"]),
                "design_reason": str(side_state["proposal"]["design_reason"]),
                "local_slope_sign": str(side_state["proposal"]["local_slope_sign"]),
                "local_slope_dx": float(side_state["proposal"]["local_slope_dx"]),
                "k_method_requested": str(side_state["proposal"]["k_method_requested"]),
                "k_method_applied": str(side_state["proposal"]["k_method_applied"]),
                "slope_at_target": float(side_state["proposal"]["slope_at_target"]),
                "slope_status": str(side_state["proposal"]["slope_status"]),
                "k_fallback_reason": str(side_state["proposal"]["k_fallback_reason"]),
            },
            "crooks": {
                "delta_f": delta_f,
                "delta_f_uncertainty": delta_f_unc,
                "forward_work_mean": float(np.mean(forward_works)),
                "forward_work_std": float(np.std(forward_works)),
                "reverse_work_mean": float(np.mean(reverse_works)),
                "reverse_work_std": float(np.std(reverse_works)),
            },
            "forward": {
                "path_file": str(forward_path_file.relative_to(side_root)),
                "draws_file": str((forward_dir / "drawn_start_samples.csv").relative_to(side_root)),
                "endpoints_file": str((forward_dir / "forward_endpoints.csv").relative_to(side_root)),
                "n_traj": int(len(synthetic_trajectories)),
            },
            "reverse": {
                "path_file": str(reverse_path_file.relative_to(side_root)),
                "draws_file": str((reverse_dir / "drawn_start_samples.csv").relative_to(side_root)),
                "endpoints_file": str((reverse_dir / "reverse_endpoints.csv").relative_to(side_root)),
                "n_traj": int(len(reverse_trajectories)),
            },
            "pmf": {
                "pmf_file": "reversible_pmf.csv",
            },
        }
        if side_state.get("truncation_metadata"):
            side_summary["truncation"] = dict(side_state["truncation_metadata"])
        write_json(side_root / "side_summary.json", side_summary)
        return side_summary

    def write_current_pair_segment_from_states(
        generation_idx: int,
        pair_root_name: str,
        left_pair_window: dict,
        right_pair_window: dict,
        left_pair_state: dict,
        right_pair_state: dict,
    ) -> Path:
        pair_root = generations_root / f"g{generation_idx:02d}" / pair_root_name
        if float(left_pair_window["center_x"]) <= float(right_pair_window["center_x"]):
            ordered_left_window = left_pair_window
            ordered_right_window = right_pair_window
            ordered_forward_state = left_pair_state
            ordered_reverse_state = right_pair_state
        else:
            ordered_left_window = right_pair_window
            ordered_right_window = left_pair_window
            ordered_forward_state = right_pair_state
            ordered_reverse_state = left_pair_state
        write_mines_bidirectional_segment_from_saved_paths(
            ctx=ctx,
            out_root=pair_root,
            left_name=str(ordered_left_window["name"]),
            right_name=str(ordered_right_window["name"]),
            left_center=float(ordered_left_window["center_x"]),
            right_center=float(ordered_right_window["center_x"]),
            left_k=float(ordered_left_window["k"]),
            right_k=float(ordered_right_window["k"]),
            left_tail_rows=ordered_left_window["tail_rows"],
            right_tail_rows=ordered_right_window["tail_rows"],
            left_tail_ref=str(ordered_left_window["tail_file"]),
            right_tail_ref=str(ordered_right_window["tail_file"]),
            forward_path_rows=ordered_forward_state["base_path_rows"],
            reverse_path_rows=ordered_reverse_state["base_path_rows"],
            forward_draw_rows=ordered_forward_state["draw_rows"],
            forward_endpoint_rows=ordered_forward_state["endpoint_rows"],
            forward_trajectories=ordered_forward_state["trajectories"],
            reverse_draw_rows=ordered_reverse_state["draw_rows"],
            reverse_endpoint_rows=ordered_reverse_state["endpoint_rows"],
            reverse_trajectories=ordered_reverse_state["trajectories"],
        )
        return pair_root

    generation_summaries: list[dict] = []
    stop_reason = "m_max"
    merged_window_name = None
    terminal_stop_details: dict | None = None
    start_generation_idx = 0
    summary_path = out_root / "mines_current_protocol_summary.json"

    if summary_path.exists():
        existing_summary = load_json(summary_path)
        if existing_summary.get("stop_reason") == "merged":
            generation_summaries = [
                dict(row)
                for row in existing_summary.get("generations", [])
                if int(row.get("generation", -1)) == 0
            ]
            left_window = load_existing_window("L1")
            right_window = load_existing_window("R1")
            start_generation_idx = 1
            for gen_dir in sorted(generations_root.glob("g[0-9][0-9]")):
                if int(gen_dir.name[1:]) >= 1:
                    shutil.rmtree(gen_dir)
            for window_dir in sorted(windows_root.iterdir()):
                if window_dir.is_dir() and window_dir.name.lower() not in {"l0", "r0", "l1", "r1"}:
                    shutil.rmtree(window_dir)
        else:
            raise RuntimeError(
                "MiNES current protocol reuse expects an existing merged run so it can "
                "discard the merged branch and continue from the saved L1/R1 state."
            )
    else:
        left_window = make_window("L0", float(ctx["basins"]["left"]), float(params["k_min"]), None, seed + 1000)
        right_window = make_window("R0", float(ctx["basins"]["right"]), float(params["k_min"]), None, seed + 2000)

    window_state_by_name: dict[str, dict] = {
        str(left_window["name"]): left_window,
        str(right_window["name"]): right_window,
    }
    def window_name_sort_key(name: str) -> tuple[int, int, str]:
        text = str(name)
        digits = "".join(ch for ch in text if ch.isdigit())
        level = int(digits) if digits else 999
        if text.startswith("L"):
            return (0, level, text)
        if text.startswith("M"):
            return (1, level, text)
        if text.startswith("R"):
            return (2, -level, text)
        return (3, level, text)

    def mode_x_from_values(values: list[float] | np.ndarray, shared_grid: np.ndarray) -> float:
        counts = histogram_counts([float(value) for value in values], shared_grid)
        return float(shared_grid[int(np.argmax(counts))]) if len(shared_grid) else float("nan")

    def rename_window_in_place(window: dict, new_name: str) -> dict:
        old_name = str(window["name"])
        if old_name == str(new_name):
            return window
        window_root = Path(window["root"])
        summary_path_local = window_root / "window_summary.json"
        summary_local = load_json(summary_path_local)
        summary_local["name"] = str(new_name)
        summary_local["renamed_from"] = old_name
        write_json(summary_path_local, summary_local)
        window["name"] = str(new_name)
        return window

    def build_truncated_forward_state(
        side_state: dict,
        threshold_x: float,
        trigger_mode: str,
    ) -> dict:
        trajectories = list(side_state["trajectories"])
        if not trajectories:
            raise RuntimeError("Swapped-child truncation requires saved forward trajectories.")
        n_time = min(
            len(side_state["base_path_rows"]),
            *(len(traj) for traj in trajectories),
        )
        if n_time <= 0:
            raise RuntimeError("Swapped-child truncation found no shared protocol rows.")

        trigger_idx = n_time - 1
        trigger_value = float("nan")
        trigger_found = False
        for row_idx in range(n_time):
            xs = np.asarray([float(traj[row_idx]["x"]) for traj in trajectories], dtype=float)
            if trigger_mode == "max_gt":
                trigger_value = float(np.max(xs))
                if trigger_value > float(threshold_x):
                    trigger_idx = row_idx
                    trigger_found = True
                    break
            elif trigger_mode == "min_lt":
                trigger_value = float(np.min(xs))
                if trigger_value < float(threshold_x):
                    trigger_idx = row_idx
                    trigger_found = True
                    break
            else:
                raise RuntimeError(f"Unknown swapped-child trigger mode: {trigger_mode}")
        if not trigger_found:
            xs = np.asarray([float(traj[trigger_idx]["x"]) for traj in trajectories], dtype=float)
            trigger_value = float(np.max(xs) if trigger_mode == "max_gt" else np.min(xs))

        truncated_base_path_rows = [dict(row) for row in side_state["base_path_rows"][: trigger_idx + 1]]
        truncated_draw_rows = [dict(row) for row in side_state["draw_rows"]]
        truncated_trajectories: list[list[dict[str, str]]] = []
        truncated_endpoint_rows: list[dict] = []
        for draw_row, traj_rows in zip(truncated_draw_rows, trajectories):
            truncated_rows = [dict(row) for row in traj_rows[: trigger_idx + 1]]
            if not truncated_rows:
                raise RuntimeError("Swapped-child truncation produced an empty trajectory.")
            truncated_trajectories.append(truncated_rows)
            final_row = truncated_rows[-1]
            truncated_endpoint_rows.append(
                {
                    "draw_idx": int(draw_row["draw_idx"]),
                    "start_x": float(draw_row["start_x"]),
                    "start_y": float(draw_row["start_y"]),
                    "final_step": int(float(final_row["step"])),
                    "final_lambda": float(final_row["lambda"]),
                    "final_x": float(final_row["x"]),
                    "final_y": float(final_row.get("y", "0.0") or 0.0),
                    "final_work": float(final_row["work"]),
                }
            )

        truncated_state = dict(side_state)
        truncated_state["base_path_rows"] = truncated_base_path_rows
        truncated_state["trajectories"] = truncated_trajectories
        truncated_state["draw_rows"] = truncated_draw_rows
        truncated_state["endpoint_rows"] = truncated_endpoint_rows
        truncated_state["protocol_variant"] = "window_swap_prime_child"
        trigger_step = (
            int(truncated_endpoint_rows[0]["final_step"])
            if truncated_endpoint_rows
            else int(trigger_idx)
        )
        truncated_state["truncation_metadata"] = {
            "trigger_mode": str(trigger_mode),
            "threshold_x": float(threshold_x),
            "trigger_found": bool(trigger_found),
            "trigger_row_index": int(trigger_idx),
            "trigger_step": int(trigger_step),
            "trigger_center_x": float(truncated_base_path_rows[-1]["center_x"]),
            "trigger_k": float(truncated_base_path_rows[-1]["k"]),
            "trigger_value": float(trigger_value),
        }
        return truncated_state

    def nes_segment_ref(
        segment_dir: str,
        left_name: str,
        right_name: str,
        orientation: str = "forward",
    ) -> dict:
        return {
            "kind": "nes_segment",
            "segment_dir": str(segment_dir),
            "left_name": str(left_name),
            "right_name": str(right_name),
            "orientation": str(orientation),
        }

    def eq_overlap_ref(
        summary_file: str,
        left_name: str,
        right_name: str,
        orientation: str = "forward",
    ) -> dict:
        return {
            "kind": "eq_overlap_mbar",
            "summary_file": str(summary_file),
            "left_name": str(left_name),
            "right_name": str(right_name),
            "orientation": str(orientation),
        }

    def segment_protocol_path_files(segment_root: Path) -> tuple[Path, Path]:
        protocol_dir = segment_root / "protocols"
        forward_candidates = [
            protocol_dir / "forward_path.csv",
            protocol_dir / "forward_augmented_path.csv",
        ]
        reverse_path_file = protocol_dir / "reverse_path.csv"
        forward_path_file = next((path for path in forward_candidates if path.exists()), None)
        if forward_path_file is None or not reverse_path_file.exists():
            raise RuntimeError(
                f"MiNES rescue step expected saved protocol paths under {segment_root}."
            )
        return forward_path_file, reverse_path_file

    used_budget_steps = float(2.0 * eq_stage_cost)
    base_stop_mode = str(stop_reason)
    base_active_segment_refs: list[dict] = []
    active_segment_refs: list[dict] = []
    base_stitched_segment_refs: list[dict] = []
    active_stitched_segment_refs: list[dict] = []
    rescue_steps: list[dict] = []

    for generation_idx in range(start_generation_idx, int(params["m_max"])):
        pair_js = js_divergence(
            [float(row["x"]) for row in left_window["tail_rows"]],
            [float(row["x"]) for row in right_window["tail_rows"]],
            grid,
        )
        pair_overlap_coeff = overlap_coefficient(
            [float(row["x"]) for row in left_window["tail_rows"]],
            [float(row["x"]) for row in right_window["tail_rows"]],
            grid,
        )
        generation_record = {
            "generation": int(generation_idx),
            "merged": False,
            "left_parent": left_window["name"],
            "right_parent": right_window["name"],
            "left_child": "",
            "right_child": "",
            "left_proposed_child": "",
            "right_proposed_child": "",
            "left_target_x": float("nan"),
            "right_target_x": float("nan"),
            "left_side_dir": "",
            "right_side_dir": "",
            "left_child_x_most": float("nan"),
            "right_child_x_most": float("nan"),
            "current_pair_js_divergence": float(pair_js),
            "current_pair_js_threshold": float(params["js_divergence_stop"]),
            "current_pair_overlap_coefficient": float(pair_overlap_coeff),
            "left_x_most_eop": float("nan"),
            "right_x_most_eop": float("nan"),
            "left_x_max_eop": float("nan"),
            "right_x_min_eop": float("nan"),
            "eop_crossed": False,
            "continue_chain_growth": False,
            "stop_event": "",
        }

        if pair_js <= float(params["js_divergence_stop"]):
            overlap_summary = write_overlap_mbar_summary(
                generation_idx,
                left_window,
                right_window,
                pair_js,
                mode="eq_overlap_mbar",
                root_name="eq_overlap_mbar",
            )
            generation_record["stop_event"] = "eq_overlap_mbar"
            generation_summaries.append(generation_record)
            stop_reason = "eq_overlap_mbar"
            base_stop_mode = "eq_overlap_mbar"
            base_active_segment_refs = [
                eq_overlap_ref(
                    str(
                        (
                            generations_root
                            / f"g{generation_idx:02d}"
                            / "eq_overlap_mbar"
                            / "overlap_summary.json"
                        ).relative_to(out_root)
                    ),
                    str(left_window["name"]),
                    str(right_window["name"]),
                )
            ]
            active_segment_refs = list(base_active_segment_refs)
            terminal_stop_details = {
                "generation": int(generation_idx),
                "mode": "eq_overlap_mbar",
                "summary_dir": str((generations_root / f"g{generation_idx:02d}" / "eq_overlap_mbar").relative_to(out_root)),
                "summary_file": str(
                    (
                        generations_root
                        / f"g{generation_idx:02d}"
                        / "eq_overlap_mbar"
                        / "overlap_summary.json"
                    ).relative_to(out_root)
                ),
                "left_window": str(left_window["name"]),
                "right_window": str(right_window["name"]),
                "js_divergence": float(pair_js),
                "threshold": float(params["js_divergence_stop"]),
                "overlap_coefficient": float(overlap_summary["overlap_coefficient"]),
                "base_mode": "eq_overlap_mbar",
            }
            break

        left_state = None
        right_state = None
        if start_generation_idx > 0 and generation_idx == start_generation_idx:
            left_state = load_existing_forward_generation_side(generation_idx, "left", left_window, right_window)
            right_state = load_existing_forward_generation_side(generation_idx, "right", right_window, left_window)
        if left_state is None:
            left_state = run_forward_generation_side(
                generation_idx,
                "left",
                left_window,
                right_window,
                10000 + 10000 * generation_idx,
            )
        if right_state is None:
            right_state = run_forward_generation_side(
                generation_idx,
                "right",
                right_window,
                left_window,
                20000 + 10000 * generation_idx,
            )
        used_budget_steps += float(side_pair_cost)

        left_eop_x_most = mode_x_from_values(
            [float(row["final_x"]) for row in left_state["endpoint_rows"]],
            grid,
        )
        right_eop_x_most = mode_x_from_values(
            [float(row["final_x"]) for row in right_state["endpoint_rows"]],
            grid,
        )
        left_eop_x_max = float(
            np.max(np.asarray([float(row["final_x"]) for row in left_state["endpoint_rows"]], dtype=float))
        )
        right_eop_x_min = float(
            np.min(np.asarray([float(row["final_x"]) for row in right_state["endpoint_rows"]], dtype=float))
        )
        eop_crossed = float(left_eop_x_max) > float(right_eop_x_min)
        generation_record.update(
            {
                "left_side_dir": str((generations_root / f"g{generation_idx:02d}" / "left").relative_to(out_root)),
                "right_side_dir": str((generations_root / f"g{generation_idx:02d}" / "right").relative_to(out_root)),
                "left_x_most_eop": float(left_eop_x_most),
                "right_x_most_eop": float(right_eop_x_most),
                "left_x_max_eop": float(left_eop_x_max),
                "right_x_min_eop": float(right_eop_x_min),
                "eop_crossed": bool(eop_crossed),
                "continue_chain_growth": bool(not eop_crossed),
            }
        )

        if eop_crossed:
            current_pair_root = write_current_pair_segment_from_states(
                generation_idx,
                "current_pair",
                left_window,
                right_window,
                left_state,
                right_state,
            )
            generation_record.update(
                {
                    "stop_event": "eop_cross_pair",
                    "current_pair_dir": str(current_pair_root.relative_to(out_root)),
                    "current_pair_summary_file": str((current_pair_root / "segment_summary.json").relative_to(out_root)),
                }
            )
            generation_summaries.append(generation_record)
            stop_reason = "eop_cross_pair"
            base_stop_mode = "eop_cross_pair"
            base_active_segment_refs = [
                nes_segment_ref(
                    str(current_pair_root.relative_to(out_root)),
                    str(left_window["name"]),
                    str(right_window["name"]),
                )
            ]
            active_segment_refs = list(base_active_segment_refs)
            terminal_stop_details = {
                "generation": int(generation_idx),
                "mode": "eop_cross_pair",
                "base_mode": "eop_cross_pair",
                "current_pair_dir": str(current_pair_root.relative_to(out_root)),
                "current_pair_summary_file": str((current_pair_root / "segment_summary.json").relative_to(out_root)),
                "left_window": str(left_window["name"]),
                "right_window": str(right_window["name"]),
                "left_x_most_eop": float(left_eop_x_most),
                "right_x_most_eop": float(right_eop_x_most),
                "left_x_max_eop": float(left_eop_x_max),
                "right_x_min_eop": float(right_eop_x_min),
                "end_of_grow_rule": "x_max_eop_left_gt_x_min_eop_right",
            }
            break

        left_target = float(left_state["proposal"]["target_x"])
        right_target = float(right_state["proposal"]["target_x"])
        left_selection = select_low_work_start_candidate(
            collect_saved_neq_candidates(left_state["trajectories"], "left"),
            left_target,
            grid,
        )
        right_selection = select_low_work_start_candidate(
            collect_saved_neq_candidates(right_state["trajectories"], "right"),
            right_target,
            grid,
        )
        next_left_window = make_window(
            f"L{generation_idx + 1}",
            float(left_state["proposal"]["center_x"]),
            float(left_state["proposal"]["k"]),
            (float(left_selection["x"]), float(left_selection["y"])),
            seed + 50000 + generation_idx,
        )
        next_right_window = make_window(
            f"R{generation_idx + 1}",
            float(right_state["proposal"]["center_x"]),
            float(right_state["proposal"]["k"]),
            (float(right_selection["x"]), float(right_selection["y"])),
            seed + 60000 + generation_idx,
        )
        child_windows_swapped = float(next_right_window["x_most"]) < float(next_left_window["x_most"])
        prime_pair_root = None
        if child_windows_swapped:
            next_left_window = rename_window_in_place(next_left_window, f"R'{generation_idx + 1}")
            next_right_window = rename_window_in_place(next_right_window, f"L'{generation_idx + 1}")
            next_left_active_window = next_right_window
            next_right_active_window = next_left_window
            left_truncated_state = build_truncated_forward_state(
                left_state,
                float(next_left_active_window["x_most"]),
                "max_gt",
            )
            right_truncated_state = build_truncated_forward_state(
                right_state,
                float(next_right_active_window["x_most"]),
                "min_lt",
            )
            left_side_summary = finalize_generation_side(
                generation_idx,
                left_truncated_state,
                next_left_active_window,
                right_selection,
                50000 + 10000 * generation_idx,
            )
            right_side_summary = finalize_generation_side(
                generation_idx,
                right_truncated_state,
                next_right_active_window,
                left_selection,
                60000 + 10000 * generation_idx,
            )
            prime_pair_root = generations_root / f"g{generation_idx:02d}" / "prime_pair"
            run_mines_bidirectional_segment_generic(
                system_root=system_root,
                ctx=ctx,
                seed=seed,
                bin_path=bin_path,
                out_root=prime_pair_root,
                left_name=str(next_left_active_window["name"]),
                right_name=str(next_right_active_window["name"]),
                left_center=float(next_left_active_window["center_x"]),
                right_center=float(next_right_active_window["center_x"]),
                left_k=float(next_left_active_window["k"]),
                right_k=float(next_right_active_window["k"]),
                left_tail_rows=next_left_active_window["tail_rows"],
                right_tail_rows=next_right_active_window["tail_rows"],
                left_tail_ref=str(next_left_active_window["tail_file"]),
                right_tail_ref=str(next_right_active_window["tail_file"]),
                t_nes=int(params["t_neq"]),
                n_nes=int(params["n_neq_traj"]),
                use_tail=float(params["tail_fraction"]),
                seed_offset=70000 + 10000 * generation_idx,
            )
            used_budget_steps += float(side_pair_cost + generic_pair_cost + 2.0 * eq_stage_cost)
        else:
            next_left_active_window = next_left_window
            next_right_active_window = next_right_window
            left_side_summary = finalize_generation_side(
                generation_idx,
                left_state,
                next_left_active_window,
                left_selection,
                50000 + 10000 * generation_idx,
            )
            right_side_summary = finalize_generation_side(
                generation_idx,
                right_state,
                next_right_active_window,
                right_selection,
                60000 + 10000 * generation_idx,
            )
            used_budget_steps += float(side_pair_cost + 2.0 * eq_stage_cost)

        child_js = js_divergence(
            [float(row["x"]) for row in next_left_active_window["tail_rows"]],
            [float(row["x"]) for row in next_right_active_window["tail_rows"]],
            grid,
        )
        child_overlap_coeff = overlap_coefficient(
            [float(row["x"]) for row in next_left_active_window["tail_rows"]],
            [float(row["x"]) for row in next_right_active_window["tail_rows"]],
            grid,
        )
        generation_record.update(
            {
                "left_child": next_left_active_window["name"],
                "right_child": next_right_active_window["name"],
                "left_proposed_child": f"L{generation_idx + 1}",
                "right_proposed_child": f"R{generation_idx + 1}",
                "left_target_x": left_target,
                "right_target_x": right_target,
                "left_child_x_most": float(next_left_active_window["x_most"]),
                "right_child_x_most": float(next_right_active_window["x_most"]),
                "child_js_divergence": float(child_js),
                "child_js_threshold": float(params["js_divergence_stop"]),
                "child_overlap_coefficient": float(child_overlap_coeff),
                "stop_event": "continue_chain_growth_window_swap" if child_windows_swapped else "continue_chain_growth",
                "left_side_summary_file": str((Path(generation_record["left_side_dir"]) / "side_summary.json")),
                "right_side_summary_file": str((Path(generation_record["right_side_dir"]) / "side_summary.json")),
                "child_windows_swapped": bool(child_windows_swapped),
            }
        )
        if child_windows_swapped and prime_pair_root is not None:
            generation_record["prime_pair_dir"] = str(prime_pair_root.relative_to(out_root))
            generation_record["prime_pair_summary_file"] = str((prime_pair_root / "segment_summary.json").relative_to(out_root))
        generation_summaries.append(generation_record)
        window_state_by_name[str(next_left_active_window["name"])] = next_left_active_window
        window_state_by_name[str(next_right_active_window["name"])] = next_right_active_window
        left_window = next_left_active_window
        right_window = next_right_active_window

    if terminal_stop_details is None:
        final_generation_idx = int(params["m_max"])
        final_pair_js = js_divergence(
            [float(row["x"]) for row in left_window["tail_rows"]],
            [float(row["x"]) for row in right_window["tail_rows"]],
            grid,
        )
        final_pair_overlap_coeff = overlap_coefficient(
            [float(row["x"]) for row in left_window["tail_rows"]],
            [float(row["x"]) for row in right_window["tail_rows"]],
            grid,
        )
        final_generation_record = {
            "generation": int(final_generation_idx),
            "merged": False,
            "left_parent": left_window["name"],
            "right_parent": right_window["name"],
            "left_child": "",
            "right_child": "",
            "left_proposed_child": "",
            "right_proposed_child": "",
            "left_target_x": float("nan"),
            "right_target_x": float("nan"),
            "left_side_dir": "",
            "right_side_dir": "",
            "left_child_x_most": float("nan"),
            "right_child_x_most": float("nan"),
            "current_pair_js_divergence": float(final_pair_js),
            "current_pair_js_threshold": float(params["js_divergence_stop"]),
            "current_pair_overlap_coefficient": float(final_pair_overlap_coeff),
            "left_x_most_eop": float("nan"),
            "right_x_most_eop": float("nan"),
            "eop_crossed": False,
            "continue_chain_growth": False,
            "stop_event": "",
        }
        if final_pair_js <= float(params["js_divergence_stop"]):
            overlap_summary = write_overlap_mbar_summary(
                final_generation_idx,
                left_window,
                right_window,
                final_pair_js,
                mode="eq_overlap_mbar",
                root_name="eq_overlap_mbar",
            )
            final_generation_record["stop_event"] = "eq_overlap_mbar"
            generation_summaries.append(final_generation_record)
            stop_reason = "eq_overlap_mbar"
            base_stop_mode = "eq_overlap_mbar"
            base_active_segment_refs = [
                eq_overlap_ref(
                    str(
                        (
                            generations_root
                            / f"g{final_generation_idx:02d}"
                            / "eq_overlap_mbar"
                            / "overlap_summary.json"
                        ).relative_to(out_root)
                    ),
                    str(left_window["name"]),
                    str(right_window["name"]),
                )
            ]
            active_segment_refs = list(base_active_segment_refs)
            terminal_stop_details = {
                "generation": int(final_generation_idx),
                "mode": "eq_overlap_mbar",
                "base_mode": "eq_overlap_mbar",
                "summary_dir": str((generations_root / f"g{final_generation_idx:02d}" / "eq_overlap_mbar").relative_to(out_root)),
                "summary_file": str(
                    (
                        generations_root
                        / f"g{final_generation_idx:02d}"
                        / "eq_overlap_mbar"
                        / "overlap_summary.json"
                    ).relative_to(out_root)
                ),
                "left_window": str(left_window["name"]),
                "right_window": str(right_window["name"]),
                "js_divergence": float(final_pair_js),
                "threshold": float(params["js_divergence_stop"]),
                "overlap_coefficient": float(overlap_summary["overlap_coefficient"]),
            }
        else:
            final_left_state = run_forward_generation_side(
                final_generation_idx,
                "left",
                left_window,
                right_window,
                10000 + 10000 * final_generation_idx,
            )
            final_right_state = run_forward_generation_side(
                final_generation_idx,
                "right",
                right_window,
                left_window,
                20000 + 10000 * final_generation_idx,
            )
            used_budget_steps += float(side_pair_cost)
            final_left_eop_x_most = mode_x_from_values(
                [float(row["final_x"]) for row in final_left_state["endpoint_rows"]],
                grid,
            )
            final_right_eop_x_most = mode_x_from_values(
                [float(row["final_x"]) for row in final_right_state["endpoint_rows"]],
                grid,
            )
            final_pair_root = write_current_pair_segment_from_states(
                final_generation_idx,
                "current_pair",
                left_window,
                right_window,
                final_left_state,
                final_right_state,
            )
            final_eop_crossed = float(final_left_eop_x_most) >= float(final_right_eop_x_most)
            final_generation_record.update(
                {
                    "left_side_dir": str((generations_root / f"g{final_generation_idx:02d}" / "left").relative_to(out_root)),
                    "right_side_dir": str((generations_root / f"g{final_generation_idx:02d}" / "right").relative_to(out_root)),
                    "left_x_most_eop": float(final_left_eop_x_most),
                    "right_x_most_eop": float(final_right_eop_x_most),
                    "eop_crossed": bool(final_eop_crossed),
                    "continue_chain_growth": False,
                    "stop_event": "eop_cross_pair" if final_eop_crossed else "m_max_current_pair",
                    "current_pair_dir": str(final_pair_root.relative_to(out_root)),
                    "current_pair_summary_file": str((final_pair_root / "segment_summary.json").relative_to(out_root)),
                }
            )
            generation_summaries.append(final_generation_record)
            stop_reason = "eop_cross_pair" if final_eop_crossed else "m_max_current_pair"
            base_stop_mode = str(stop_reason)
            base_active_segment_refs = [
                nes_segment_ref(
                    str(final_pair_root.relative_to(out_root)),
                    str(left_window["name"]),
                    str(right_window["name"]),
                )
            ]
            active_segment_refs = list(base_active_segment_refs)
            terminal_stop_details = {
                "generation": int(final_generation_idx),
                "mode": str(stop_reason),
                "base_mode": str(stop_reason),
                "current_pair_dir": str(final_pair_root.relative_to(out_root)),
                "current_pair_summary_file": str((final_pair_root / "segment_summary.json").relative_to(out_root)),
                "left_window": str(left_window["name"]),
                "right_window": str(right_window["name"]),
                "left_x_most_eop": float(final_left_eop_x_most),
                "right_x_most_eop": float(final_right_eop_x_most),
            }

    if base_active_segment_refs:
        active_segment_refs = [dict(row) for row in base_active_segment_refs]
    elif terminal_stop_details.get("summary_file"):
        active_segment_refs = [
            eq_overlap_ref(
                str(terminal_stop_details["summary_file"]),
                str(terminal_stop_details.get("left_window", "")),
                str(terminal_stop_details.get("right_window", "")),
            )
        ]

    growth_generation_rows = [
        row
        for row in generation_summaries
        if str(row.get("left_side_dir", ""))
        and str(row.get("right_side_dir", ""))
        and str(row.get("left_child", ""))
        and str(row.get("right_child", ""))
    ]
    left_stitched_segment_refs = [
        nes_segment_ref(
            str(row["left_side_dir"]),
            str(row["left_parent"]),
            str(row["left_child"]),
            orientation="forward",
        )
        for row in sorted(growth_generation_rows, key=lambda item: int(item["generation"]))
    ]
    right_stitched_segment_refs = [
        nes_segment_ref(
            str(row["right_side_dir"]),
            str(row["right_child"]),
            str(row["right_parent"]),
            orientation="reverse",
        )
        for row in sorted(growth_generation_rows, key=lambda item: int(item["generation"]), reverse=True)
    ]
    middle_stitched_segment_refs = [dict(row) for row in base_active_segment_refs]
    if not middle_stitched_segment_refs and terminal_stop_details.get("summary_file"):
        middle_stitched_segment_refs = [
            eq_overlap_ref(
                str(terminal_stop_details["summary_file"]),
                str(terminal_stop_details.get("left_window", "")),
                str(terminal_stop_details.get("right_window", "")),
            )
        ]
    base_stitched_segment_refs = (
        [dict(row) for row in left_stitched_segment_refs]
        + [dict(row) for row in middle_stitched_segment_refs]
        + [dict(row) for row in right_stitched_segment_refs]
    )
    if base_stitched_segment_refs:
        active_stitched_segment_refs = [dict(row) for row in base_stitched_segment_refs]
    elif active_segment_refs:
        active_stitched_segment_refs = [dict(row) for row in active_segment_refs]

    rescue_step_cost = float(eq_stage_cost + 2.0 * generic_pair_cost)
    if active_stitched_segment_refs and stop_reason in {"eop_cross_pair", "eq_overlap_mbar"}:
        rescue_iteration = 0
        while used_budget_steps < total_budget_steps - 1.0e-9:
            candidate_entries = [
                entry
                for entry in active_stitched_segment_refs
                if str(entry.get("kind", "")) == "nes_segment"
            ]
            if not candidate_entries:
                break

            ranked_candidates = []
            for entry in candidate_entries:
                left_candidate_window = window_state_by_name.get(str(entry.get("left_name", "")))
                right_candidate_window = window_state_by_name.get(str(entry.get("right_name", "")))
                segment_root = out_root / str(entry.get("segment_dir", ""))
                if left_candidate_window is None or right_candidate_window is None or not segment_root.exists():
                    continue
                try:
                    candidate = analyze_rescue_candidate(
                        int(terminal_stop_details.get("generation", -1)),
                        rescue_iteration + 1,
                        left_candidate_window,
                        right_candidate_window,
                        segment_root,
                    )
                except RuntimeError:
                    continue
                ranked_candidates.append((entry, candidate))
            if not ranked_candidates:
                break

            ranked_candidates.sort(
                key=lambda item: rescue_candidate_priority_key(item[1]),
                reverse=True,
            )
            base_entry, best_candidate = ranked_candidates[0]
            left_candidate_window = window_state_by_name[str(base_entry["left_name"])]
            right_candidate_window = window_state_by_name[str(base_entry["right_name"])]
            base_segment_root = out_root / str(base_entry["segment_dir"])
            rescue_pair_js = js_divergence(
                [float(row["x"]) for row in left_candidate_window["tail_rows"]],
                [float(row["x"]) for row in right_candidate_window["tail_rows"]],
                grid,
            )
            if rescue_pair_js <= float(params["js_divergence_stop"]):
                rescue_result = write_rescue_overlap_mbar_step(
                    int(terminal_stop_details.get("generation", -1)),
                    rescue_iteration + 1,
                    left_candidate_window,
                    right_candidate_window,
                    base_segment_root,
                    rescue_pair_js,
                )
                replacement_refs = [
                    eq_overlap_ref(
                        str(rescue_result["summary"]["overlap_summary_file"]),
                        str(left_candidate_window["name"]),
                        str(right_candidate_window["name"]),
                        orientation="forward",
                    )
                ]
            else:
                rescue_result = write_rescue_step(
                    int(terminal_stop_details.get("generation", -1)),
                    rescue_iteration + 1,
                    left_candidate_window,
                    right_candidate_window,
                    base_segment_root,
                    rescue_pair_js,
                    candidate=best_candidate,
                )
                used_budget_steps += float(rescue_step_cost)
                rescue_window = rescue_result["window"]
                window_state_by_name[str(rescue_window["name"])] = rescue_window
                replacement_refs = [
                    nes_segment_ref(
                        str(rescue_result["left_segment_root"].relative_to(out_root)),
                        str(left_candidate_window["name"]),
                        str(rescue_window["name"]),
                        orientation="forward",
                    ),
                    nes_segment_ref(
                        str(rescue_result["right_segment_root"].relative_to(out_root)),
                        str(rescue_window["name"]),
                        str(right_candidate_window["name"]),
                        orientation="forward",
                    ),
                ]
            replace_idx = next(
                (
                    idx
                    for idx, entry in enumerate(active_stitched_segment_refs)
                    if str(entry.get("kind", "")) == "nes_segment"
                    and str(entry.get("segment_dir", "")) == str(base_entry["segment_dir"])
                ),
                None,
            )
            if replace_idx is None:
                active_stitched_segment_refs.extend(replacement_refs)
            else:
                active_stitched_segment_refs[replace_idx : replace_idx + 1] = replacement_refs

            rescue_result["summary"]["active_segment_refs_after_step"] = [
                dict(entry) for entry in active_stitched_segment_refs
            ]
            rescue_result["summary"]["active_stitched_segment_refs_after_step"] = [
                dict(entry) for entry in active_stitched_segment_refs
            ]
            rescue_result["summary"]["base_segment_ref"] = dict(base_entry)
            write_json(rescue_result["rescue_root"] / "rescue_summary.json", rescue_result["summary"])
            rescue_steps.append(dict(rescue_result["summary"]))
            rescue_iteration += 1

        if rescue_steps:
            stop_reason = "rescue_window"
            terminal_stop_details["mode"] = "rescue_window"

    terminal_stop_details["base_mode"] = str(base_stop_mode)
    terminal_stop_details["base_active_segment_refs"] = [dict(row) for row in base_active_segment_refs]
    terminal_stop_details["active_segment_refs"] = [dict(row) for row in active_segment_refs]
    terminal_stop_details["base_stitched_segment_refs"] = [dict(row) for row in base_stitched_segment_refs]
    terminal_stop_details["active_stitched_segment_refs"] = [
        dict(row) for row in active_stitched_segment_refs
    ]
    terminal_stop_details["rescue_steps"] = rescue_steps
    terminal_stop_details["active_rescue_segment_dirs"] = [
        str(row.get("segment_dir", ""))
        for row in active_stitched_segment_refs
        if str(row.get("kind", "")) == "nes_segment"
    ]
    terminal_stop_details["rescue_iteration_count"] = int(len(rescue_steps))
    terminal_stop_details["refine_eq_rounds"] = []
    terminal_stop_details["budget_step_limit"] = float(total_budget_steps)
    terminal_stop_details["budget_steps_used"] = float(used_budget_steps)
    terminal_stop_details["budget_steps_remaining"] = float(max(total_budget_steps - used_budget_steps, 0.0))
    terminal_stop_details["budget_steps_overrun"] = float(max(used_budget_steps - total_budget_steps, 0.0))

    write_json(
        out_root / "mines_current_protocol_summary.json",
        {
            "label": label,
            "seed": int(seed),
            "system_root": str(system_root),
            "parameters": params,
            "stop_reason": stop_reason,
            "merged_window_name": merged_window_name,
            "generation_count": int(len(generation_summaries)),
            "budget_step_limit": float(total_budget_steps),
            "budget_steps_used": float(used_budget_steps),
            "budget_steps_remaining": float(max(total_budget_steps - used_budget_steps, 0.0)),
            "budget_steps_overrun": float(max(used_budget_steps - total_budget_steps, 0.0)),
            "refine_eq_round_count": int(len(terminal_stop_details.get("refine_eq_rounds", []))),
            "generations": generation_summaries,
            "terminal_stop_details": terminal_stop_details,
        },
    )


def main() -> None:
    args = parse_args()
    system_root = Path(args.system_root).resolve()
    if args.command == "run-aus-seed":
        run_aus_seed(system_root, args.combo_label, args.seed, args.bin)
        return
    if args.command == "run-mines-seed":
        run_mines_seed(system_root, args.combo_label, args.seed, args.bin)
        return
    if args.command == "run-mines-current-protocol":
        run_mines_current_protocol(
            system_root,
            args.seed,
            args.bin,
            args.label,
            args.t_neq,
            args.n_neq_traj,
            args.total_budget_steps,
        )
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
