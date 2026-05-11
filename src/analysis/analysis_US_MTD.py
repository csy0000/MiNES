#!/usr/bin/env python3
"""Stream-reduce raw benchmark outputs into PMF-vs-time comparisons.

This module is the analysis hub for the 1D benchmark. It reconstructs PMFs for
US, AUS, NES, MINES, and WT-MTD, writes compact reduced artifacts, generates
method GIFs, and selects the benchmark-facing winner for each method screen.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = REPO_ROOT / ".matplotlib-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
# Keep Matplotlib cache files inside the repo so plotting works the same on
# shared machines and in headless runs.
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt


RMSE_SUCCESS_THRESHOLD = 0.5
COVERAGE_SUCCESS_THRESHOLD = 0.95
GIF_SECONDS = 5.0
GIF_FRAMES = 60
REDUCED_SAMPLE_COUNT = 1000
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
                "pymbar is required for benchmark MBAR reconstruction. "
                "Install it in the active Python environment before running the reducer."
            ) from exc
        _PYMBAR = pymbar_module
    return _PYMBAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_us = subparsers.add_parser("process-us-seed")
    process_us.add_argument("--system-root", required=True)
    process_us.add_argument("--combo-label", required=True)
    process_us.add_argument("--seed", type=int, required=True)
    process_us.add_argument("--make-gif", action="store_true")

    process_aus = subparsers.add_parser("process-aus-seed")
    process_aus.add_argument("--system-root", required=True)
    process_aus.add_argument("--combo-label", required=True)
    process_aus.add_argument("--seed", type=int, required=True)

    process_mtd = subparsers.add_parser("process-mtd-seed")
    process_mtd.add_argument("--system-root", required=True)
    process_mtd.add_argument("--combo-label", required=True)
    process_mtd.add_argument("--seed", type=int, required=True)
    process_mtd.add_argument("--make-gif", action="store_true")

    process_nes = subparsers.add_parser("process-nes-seed-time")
    process_nes.add_argument("--system-root", required=True)
    process_nes.add_argument("--combo-label", required=True)
    process_nes.add_argument("--seed", type=int, required=True)
    process_nes.add_argument("--time-steps", type=int, required=True)
    process_nes.add_argument("--make-gif", action="store_true")
    process_nes.add_argument("--retain-reduced", action="store_true")

    process_mines = subparsers.add_parser("process-mines-seed")
    process_mines.add_argument("--system-root", required=True)
    process_mines.add_argument("--combo-label", required=True)
    process_mines.add_argument("--seed", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--system-root", required=True)

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_method_dat(
    path: Path,
    xs: list[float],
    snapshots: list[tuple[float, list[float | None], list[float | None]]],
    overwrite: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if overwrite or path.stat().st_size == 0:
            writer.writerow(["time", "x", "F_est", "var_est"])
        for time_value, mean_values, var_values in snapshots:
            for x, mean_value, var_value in zip(xs, mean_values, var_values):
                writer.writerow(
                    [
                        f"{time_value:.10f}",
                        f"{x:.10f}",
                        "" if mean_value is None else f"{mean_value:.10f}",
                        "" if var_value is None else f"{var_value:.10f}",
                    ]
                )


def read_method_dat(path: Path) -> list[tuple[float, list[float | None], list[float | None]]]:
    rows = read_csv_dicts(path)
    if not rows:
        return []
    times = sorted({float(row["time"]) for row in rows})
    xs = sorted({float(row["x"]) for row in rows})
    snapshots: list[tuple[float, list[float | None], list[float | None]]] = []
    for time_value in times:
        mean_values: list[float | None] = []
        var_values: list[float | None] = []
        row_map = {
            float(row["x"]): row
            for row in rows
            if abs(float(row["time"]) - time_value) < 1e-9
        }
        for x in xs:
            row = row_map[x]
            mean_values.append(None if row["F_est"] == "" else float(row["F_est"]))
            var_values.append(None if row["var_est"] == "" else float(row["var_est"]))
        snapshots.append((time_value, mean_values, var_values))
    return snapshots


def normalize_context(ctx: dict) -> dict:
    # Backfill newer benchmark keys so the reducer can still read older
    # `run_context.json` files created before the current screen layout.
    if "time_grid" not in ctx:
        values = np.geomspace(1.0e4, 1.0e7, 21)
        steps = [int(round(v / 100.0) * 100.0) for v in values]
        ctx["time_grid"] = {
            "values": steps,
            "labels": [time_label(v) for v in steps],
            "kind": "logspace_steps",
            "t_min": steps[0],
            "t_max": steps[-1],
            "count": len(steps),
        }
    if "labels" not in ctx["time_grid"]:
        ctx["time_grid"]["labels"] = [time_label(v) for v in ctx["time_grid"]["values"]]
    if "combo_labeling" not in ctx:
        ctx["combo_labeling"] = {
            "us": "k_<k>__dx_<dx> with decimal points replaced by p",
            "aus": "qnext_<q>__alpha_<a>__fit_<fit_method>__kmin_<kmin>__kmax_<kmax> with decimal points replaced by p",
            "nes": "k_<k> with decimal points replaced by p",
            "mines": "k_pull_<k> with decimal points replaced by p",
            "mtd": "biasfactor_<biasfactor> with decimal points replaced by p",
        }
    else:
        ctx["combo_labeling"].setdefault("aus", "qnext_<q>__alpha_<a>__fit_<fit_method>__kmin_<kmin>__kmax_<kmax> with decimal points replaced by p")
        ctx["combo_labeling"].setdefault("mines", "k_pull_<k> with decimal points replaced by p")
    if "mtd_screen" not in ctx and "mtd" in ctx:
        mtd = ctx["mtd"]
        ctx["mtd_screen"] = {
            "biasfactor_values": [mtd["biasfactor"]],
            "fixed": {
                "total_steps": mtd["total_steps"],
                "per_walker_steps": mtd["per_walker_steps"],
                "sample_stride_steps": mtd["sample_stride_steps"],
                "meta_nout": mtd["meta_nout"],
                "w0": mtd["w0"],
                "sigma": mtd["sigma"],
                "stride": mtd["stride"],
            },
            "combos": [
                {
                    "label": "default",
                    "biasfactor": mtd["biasfactor"],
                    "total_steps": mtd["total_steps"],
                    "per_walker_steps": mtd["per_walker_steps"],
                    "sample_stride_steps": mtd["sample_stride_steps"],
                    "meta_nout": mtd["meta_nout"],
                    "w0": mtd["w0"],
                    "sigma": mtd["sigma"],
                    "stride": mtd["stride"],
                }
            ],
        }
    ctx.setdefault("aus_screen", {"alpha_values": [], "fixed": {}, "combos": []})
    ctx.setdefault("mines_screen", {"k_pull_values": [], "fixed": {}, "combos": []})
    if "rmse_eval_grid" not in ctx:
        ctx["rmse_eval_grid"] = {
            "xmin": -10.0,
            "xmax": 10.0,
            "dx": 0.2,
        }
    return ctx


def build_grid(xmin: float, xmax: float, dx: float) -> list[float]:
    n = int(round((xmax - xmin) / dx))
    return [xmin + dx * i for i in range(n + 1)]


def grid_index(xs: list[float], dx: float, value: float) -> int:
    idx = int(math.floor((value - xs[0]) / dx + 0.5))
    if idx < 0 or idx >= len(xs):
        return -1
    return idx


def shift_to_zero(values: list[float | None]) -> list[float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return values
    shift = min(finite)
    return [None if value is None or not math.isfinite(value) else value - shift for value in values]


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def sample_variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return sum((value - mu) ** 2 for value in values) / float(len(values) - 1)


def time_label(steps: int) -> str:
    return f"T_{steps}"


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


def quad_mid_scaling_k(lamb: float, mid_scale: float) -> float:
    if mid_scale == 1.0:
        return 1.0
    if mid_scale > 1.0:
        return 1.0 + mid_scale * (0.25 - (lamb - 0.5) * (lamb - 0.5))
    scale = 4.0 * (lamb - 0.5) * (lamb - 0.5)
    return scale * (1.0 - mid_scale) + mid_scale


def harmonic_bias(x: float, center: float, k: float) -> float:
    dx = x - center
    return 0.5 * k * dx * dx


def analytic_doublewell_profile(xs: list[float], ctx: dict) -> list[float]:
    pot = ctx["potential"]
    beta = 1.0 / float(ctx["thermal_kT"])
    values: list[float] = []
    for x in xs:
        u0 = pot["k0"] * (x - pot["x0"]) * (x - pot["x0"])
        u1 = pot["k1"] * (x - pot["x1"]) * (x - pot["x1"])
        log_t0 = -beta * u0
        log_t1 = -beta * u1 - pot["E1"]
        log_max = max(log_t0, log_t1)
        log_sum = log_max + math.log(math.exp(log_t0 - log_max) + math.exp(log_t1 - log_max))
        values.append(-log_sum / beta)
    shifted = shift_to_zero(values)
    return [0.0 if value is None else value for value in shifted]


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


def build_eval_grid(ctx: dict) -> list[float]:
    grid = ctx["rmse_eval_grid"]
    return build_grid(float(grid["xmin"]), float(grid["xmax"]), float(grid["dx"]))


def sample_profile_on_grid(
    source_xs: list[float],
    source_values: list[float | None],
    target_xs: list[float],
) -> list[float | None]:
    lookup = {
        round(float(x), 10): value
        for x, value in zip(source_xs, source_values)
    }
    sampled = []
    for x in target_xs:
        value = lookup.get(round(float(x), 10))
        if value is None or not math.isfinite(value):
            sampled.append(None)
        else:
            sampled.append(float(value))
    return sampled


def us_mbar_snapshot(
    window_samples: list[np.ndarray],
    window_centers: list[float],
    window_ks: list[float],
    xs: list[float],
    ctx: dict,
    initial_f: np.ndarray | None,
) -> tuple[list[float | None], np.ndarray]:
    # Estimate the unbiased PMF from the currently available umbrella samples
    # with PyMBAR. Reusing the previous window free-energy offsets as the
    # initial guess keeps the PMF-vs-time sweep numerically smoother and faster.
    nonempty = [
        (samples, center, k)
        for samples, center, k in zip(window_samples, window_centers, window_ks)
        if samples.size > 0
    ]
    if not nonempty:
        return [None] * len(xs), np.zeros(0, dtype=float)

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

    xmin = xs[0]
    dx = float(ctx["grid"]["dx"])
    indices = np.floor((samples - xmin) / dx + 0.5).astype(int)
    valid = (indices >= 0) & (indices < len(xs))
    probability = np.zeros(len(xs), dtype=float)
    np.add.at(probability, indices[valid], weights[valid])

    free_energy = np.full(len(xs), np.nan, dtype=float)
    positive = probability > 0.0
    free_energy[positive] = -float(ctx["thermal_kT"]) * np.log(probability[positive])
    finite = np.isfinite(free_energy)
    if np.any(finite):
        free_energy[finite] -= np.min(free_energy[finite])

    result: list[float | None] = []
    for value in free_energy:
        result.append(float(value) if np.isfinite(value) else None)
    return result, reduced_free


def read_hills(path: Path) -> list[dict[str, float]]:
    rows = read_csv_dicts(path)
    hills = []
    for row in rows:
        hills.append(
            {
                "step": float(row["step"]),
                "x": float(row["x"]),
                "height": float(row["height"]),
                "sigma": float(row["sigma_x"]),
            }
        )
    return hills


def add_hill_to_bias(bias: np.ndarray, xs: np.ndarray, hill: dict[str, float]) -> None:
    sigma2 = hill["sigma"] * hill["sigma"]
    denom = 2.0 * sigma2
    bias += hill["height"] * np.exp(-((xs - hill["x"]) ** 2) / denom)


def read_trajectory(path: Path) -> list[dict[str, float]]:
    rows = read_csv_dicts(path)
    result = []
    for row in rows:
        entry = {key: float(value) for key, value in row.items() if value != ""}
        result.append(entry)
    return result


def list_traj_files(rep_dir: Path, prefix: str) -> list[Path]:
    def traj_idx(path: Path) -> int:
        stem = path.stem
        return int(stem.split("_")[-1])

    return sorted(rep_dir.glob(f"{prefix}_*.csv"), key=traj_idx)


def center_from_lambda(ctx: dict, lamb: float) -> float:
    x0 = ctx["potential"]["x0"]
    x1 = ctx["potential"]["x1"]
    return x0 + lamb * (x1 - x0)


def center_from_lambda_bounds(left_x: float, right_x: float, lamb: float) -> float:
    return left_x + lamb * (right_x - left_x)


def k_from_lambda(base_k: float, mid_scale: float, lamb: float) -> float:
    return base_k * quad_mid_scaling_k(lamb, mid_scale)


def hs_reconstruct_segment(
    sum_w: list[float],
    sum_hist: list[list[float]],
    lambdas: list[float],
    xs: list[float],
    ctx: dict,
    n_traj: int,
    base_k: float,
    mid_scale: float,
    left_x: float,
    right_x: float,
) -> list[float | None]:
    log_sum_w = np.where(np.asarray(sum_w, dtype=float) > 0.0, np.log(np.asarray(sum_w, dtype=float)), -np.inf)
    hist_arr = np.asarray(sum_hist, dtype=float)
    log_sum_hist = np.where(hist_arr > 0.0, np.log(hist_arr), -np.inf)
    return hs_reconstruct_segment_from_logs(
        log_sum_w,
        log_sum_hist,
        lambdas,
        xs,
        ctx,
        n_traj,
        base_k,
        mid_scale,
        left_x,
        right_x,
    )


def hs_reconstruct_segment_from_logs(
    log_sum_w: np.ndarray,
    log_sum_hist: np.ndarray,
    lambdas: list[float],
    xs: list[float],
    ctx: dict,
    n_traj: int,
    base_k: float,
    mid_scale: float,
    left_x: float,
    right_x: float,
) -> list[float | None]:
    # Hummer-Szabo reweighting for one switching segment. The numerator carries
    # nonequilibrium work weights, while the denominator removes the moving
    # harmonic restraint.
    beta = 1.0 / float(ctx["thermal_kT"])
    log_numerator_terms = log_sum_hist - log_sum_w[:, None]
    log_denominator_terms = np.full((len(lambdas), len(xs)), -np.inf, dtype=float)
    log_n_traj = math.log(float(n_traj))
    for time_idx, lamb in enumerate(lambdas):
        if not math.isfinite(float(log_sum_w[time_idx])):
            continue
        center = center_from_lambda_bounds(left_x, right_x, lamb)
        k_value = k_from_lambda(base_k, mid_scale, lamb)
        xs_arr = np.asarray(xs, dtype=float)
        log_denominator_terms[time_idx] = (
            log_n_traj
            - float(log_sum_w[time_idx])
            - beta * 0.5 * float(k_value) * (xs_arr - center) * (xs_arr - center)
        )

    log_numerator = logsumexp_np(log_numerator_terms, axis=0)
    log_denominator = logsumexp_np(log_denominator_terms, axis=0)
    log_density = log_numerator - log_denominator
    density: list[float | None] = [None] * len(xs)
    dx = float(ctx["grid"]["dx"])
    finite = np.isfinite(log_density)
    if np.any(finite):
        log_norm = float(logsumexp_np(log_density[finite])) + math.log(dx)
        log_density[finite] -= log_norm
        for idx in np.flatnonzero(finite):
            density[int(idx)] = math.exp(float(log_density[int(idx)]))

    free_energy: list[float | None] = [None] * len(xs)
    for idx, value in enumerate(density):
        if value is None or value <= 0.0:
            continue
        free_energy[idx] = -float(ctx["thermal_kT"]) * math.log(value)
    return shift_to_zero(free_energy)


def hs_reconstruct(
    sum_w: list[float],
    sum_hist: list[list[float]],
    lambdas: list[float],
    xs: list[float],
    ctx: dict,
    n_traj: int,
    base_k: float,
    mid_scale: float,
) -> list[float | None]:
    return hs_reconstruct_segment(
        sum_w,
        sum_hist,
        lambdas,
        xs,
        ctx,
        n_traj,
        base_k,
        mid_scale,
        float(ctx["potential"]["x0"]),
        float(ctx["potential"]["x1"]),
    )


def average_densities_to_pmf(pmfs: list[list[float | None]], ctx: dict) -> list[float | None]:
    if not pmfs:
        return []
    beta = 1.0 / float(ctx["thermal_kT"])
    dx = float(ctx["grid"]["dx"])
    # Convert each PMF back to a normalized density before averaging so forward
    # and backward reconstructions contribute on the same scale.
    log_density_sum = np.full(len(pmfs[0]), -np.inf, dtype=float)
    used = 0
    for pmf in pmfs:
        log_local = np.full(len(pmf), -np.inf, dtype=float)
        finite_idx = [idx for idx, value in enumerate(pmf) if value is not None and math.isfinite(value)]
        if not finite_idx:
            continue
        log_weights = np.asarray([-beta * float(pmf[idx]) for idx in finite_idx], dtype=float)
        log_norm = float(logsumexp_np(log_weights)) + math.log(dx)
        for local_pos, idx in enumerate(finite_idx):
            log_local[idx] = float(log_weights[local_pos] - log_norm)
        log_density_sum = np.logaddexp(log_density_sum, log_local)
        used += 1

    if used <= 0:
        return [None] * len(pmfs[0])

    log_density = log_density_sum - math.log(float(used))
    finite = np.isfinite(log_density)
    if np.any(finite):
        log_norm = float(logsumexp_np(log_density[finite])) + math.log(dx)
        log_density[finite] -= log_norm

    pmf: list[float | None] = [None] * len(log_density)
    for idx in np.flatnonzero(finite):
        pmf[int(idx)] = -float(ctx["thermal_kT"]) * float(log_density[int(idx)])
    return shift_to_zero(pmf)


def combine_absolute_pmfs(pmfs: list[list[float | None]], ctx: dict) -> list[float | None]:
    if not pmfs:
        return []
    beta = 1.0 / float(ctx["thermal_kT"])
    # MINES segments are already aligned onto a common absolute scaffold, so
    # they can be combined directly in density space.
    n_bins = len(pmfs[0])
    log_density_matrix = np.full((len(pmfs), n_bins), -np.inf, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for pmf_idx, pmf in enumerate(pmfs):
        for idx, value in enumerate(pmf):
            if value is None or not math.isfinite(value):
                continue
            log_density_matrix[pmf_idx, idx] = -beta * float(value)
            counts[idx] += 1
    log_density = np.full(n_bins, -np.inf, dtype=float)
    finite_bins = counts > 0
    if np.any(finite_bins):
        log_density_sum = logsumexp_np(log_density_matrix, axis=0)
        log_density[finite_bins] = log_density_sum[finite_bins] - np.log(counts[finite_bins].astype(float))
        log_norm = float(logsumexp_np(log_density[finite_bins])) + math.log(float(ctx["grid"]["dx"]))
        log_density[finite_bins] -= log_norm
    result: list[float | None] = [None] * n_bins
    for idx in np.flatnonzero(np.isfinite(log_density)):
        result[int(idx)] = -float(ctx["thermal_kT"]) * float(log_density[int(idx)])
    return shift_to_zero(result)


def bar_delta_f(forward_works: list[float], reverse_works: list[float], beta: float) -> tuple[float, float]:
    if not forward_works or not reverse_works:
        return 0.0, math.inf

    # Solve the BAR fixed-point equation by bisection. If bracketing fails,
    # fall back to the midpoint of forward/reverse Jarzynski estimates.
    def logistic(value: float) -> float:
        if value >= 0.0:
            exp_neg = math.exp(-value)
            return exp_neg / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return 1.0 / (1.0 + exp_pos)

    def residual(delta_f: float) -> float:
        lhs = sum(logistic(beta * (work - delta_f)) for work in forward_works)
        rhs = sum(logistic(beta * (work + delta_f)) for work in reverse_works)
        return lhs - rhs

    scale = max(abs(value) for value in (forward_works + reverse_works))
    lo = -max(10.0, scale + 10.0)
    hi = max(10.0, scale + 10.0)
    flo = residual(lo)
    fhi = residual(hi)

    if flo == 0.0:
        delta_f = lo
    elif fhi == 0.0:
        delta_f = hi
    elif flo * fhi > 0.0:
        arr_f = np.asarray(forward_works, dtype=float)
        arr_r = np.asarray(reverse_works, dtype=float)
        log_mean_f = float(logsumexp_np(-beta * arr_f)) - math.log(float(len(arr_f)))
        log_mean_r = float(logsumexp_np(-beta * arr_r)) - math.log(float(len(arr_r)))
        df_f = -log_mean_f / beta
        df_r = log_mean_r / beta
        delta_f = 0.5 * (df_f + df_r)
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fmid = residual(mid)
            if abs(fmid) < 1.0e-10 or abs(hi - lo) < 1.0e-8:
                delta_f = mid
                break
            if flo * fmid <= 0.0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        else:
            delta_f = 0.5 * (lo + hi)

    uncertainty = abs(delta_f) / math.sqrt(float(len(forward_works) + len(reverse_works)))
    return float(delta_f), float(uncertainty)


def finite_values(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def compute_snapshot_rmse(
    values: list[float | None],
    analytic: list[float],
) -> float:
    errors = []
    for idx in range(len(analytic)):
        value = values[idx]
        if value is None or not math.isfinite(value):
            continue
        errors.append((value - analytic[idx]) ** 2)
    if not errors:
        return math.inf
    return math.sqrt(sum(errors) / float(len(errors)))


def compute_mean_var(var_values: list[float | None]) -> float:
    finite = []
    for idx in range(len(var_values)):
        value = var_values[idx]
        if value is not None and math.isfinite(value):
            finite.append(value)
    if not finite:
        return math.inf
    return mean(finite)


def evaluate_combo(
    snapshots: list[tuple[float, list[float | None], list[float | None]]],
    source_xs: list[float],
    eval_xs: list[float],
    analytic_eval: list[float],
) -> dict:
    if not snapshots:
        return {
            "status": "missing",
            "snapshot_count": 0,
            "final_time": math.inf,
            "central_coverage_fraction": 0.0,
            "final_rmse": math.inf,
            "final_mean_var": math.inf,
            "time_avg_rmse": math.inf,
            "time_to_rmse_le_0p5": None,
        }

    rmses = []
    time_to_threshold = None
    for time_value, mean_values, var_values in snapshots:
        eval_values = sample_profile_on_grid(source_xs, mean_values, eval_xs)
        eval_vars = sample_profile_on_grid(source_xs, var_values, eval_xs)
        coverage = len(finite_values(eval_values)) / float(len(eval_xs))
        rmse = compute_snapshot_rmse(eval_values, analytic_eval)
        if math.isfinite(rmse):
            rmses.append(rmse)
            if time_to_threshold is None and coverage >= COVERAGE_SUCCESS_THRESHOLD and rmse <= RMSE_SUCCESS_THRESHOLD:
                time_to_threshold = time_value

    final_time, final_values, final_vars = snapshots[-1]
    final_eval_values = sample_profile_on_grid(source_xs, final_values, eval_xs)
    final_eval_vars = sample_profile_on_grid(source_xs, final_vars, eval_xs)
    final_coverage = len(finite_values(final_eval_values)) / float(len(eval_xs))
    final_rmse = compute_snapshot_rmse(final_eval_values, analytic_eval)
    final_mean_var = compute_mean_var(final_eval_vars)
    time_avg_rmse = mean(rmses) if rmses else math.inf
    status = "ok" if final_coverage >= COVERAGE_SUCCESS_THRESHOLD and math.isfinite(final_rmse) else "partial"
    return {
        "status": status,
        "snapshot_count": len(snapshots),
        "final_time": final_time,
        "central_coverage_fraction": final_coverage,
        "final_rmse": final_rmse,
        "final_mean_var": final_mean_var,
        "time_avg_rmse": time_avg_rmse,
        "time_to_rmse_le_0p5": time_to_threshold,
    }


def selection_sort_key(row: dict) -> tuple[float, float, float, float, float]:
    full_coverage_penalty = 0.0 if row["central_coverage_fraction"] >= COVERAGE_SUCCESS_THRESHOLD else 1.0
    return (
        full_coverage_penalty,
        -row["central_coverage_fraction"],
        row["final_rmse"],
        row["final_mean_var"],
        row["time_avg_rmse"],
    )


def aggregate_replicates(
    replicates: list[list[tuple[float, list[float | None], list[float | None]]]]
) -> list[tuple[float, list[float | None], list[float | None]]]:
    nonempty = [rep for rep in replicates if rep]
    if not nonempty:
        return []
    snapshot_count = min(len(rep) for rep in nonempty)
    aggregated: list[tuple[float, list[float | None], list[float | None]]] = []
    # Aggregate only the common snapshot prefix so partially finished replicates
    # do not invent later-time statistics.
    for snap_idx in range(snapshot_count):
        time_value = mean([rep[snap_idx][0] for rep in nonempty])
        n_grid = len(nonempty[0][snap_idx][1])
        mean_values: list[float | None] = []
        var_values: list[float | None] = []
        for grid_idx in range(n_grid):
            finite = []
            for rep in nonempty:
                value = rep[snap_idx][1][grid_idx]
                if value is not None and math.isfinite(value):
                    finite.append(value)
            if not finite:
                mean_values.append(None)
                var_values.append(None)
                continue
            mean_values.append(mean(finite))
            var_values.append(sample_variance(finite))
        aggregated.append((time_value, shift_to_zero(mean_values), var_values))
    return aggregated


def downsample_evenly(rows: list[dict], target: int = REDUCED_SAMPLE_COUNT) -> list[dict]:
    if len(rows) <= target:
        return rows
    indices = np.linspace(0, len(rows) - 1, target, dtype=int)
    return [rows[int(idx)] for idx in indices]


def write_reduced_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_annotation(ax: plt.Axes, title: str, lines: list[str]) -> None:
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def save_gif(fig: plt.Figure, animate_fn, n_frames: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani = animation.FuncAnimation(fig, animate_fn, frames=n_frames, interval=1000.0 * GIF_SECONDS / n_frames)
    fps = max(1, int(round(n_frames / GIF_SECONDS)))
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)


def parse_float_or_none(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def load_us_windows(raw_dir: Path) -> list[dict]:
    rows = read_csv_dicts(raw_dir / "us_windows.csv")
    windows = []
    for row in sorted(rows, key=lambda item: int(item["window_id"])):
        traj_path = Path(row["traj_file"])
        if not traj_path.is_absolute():
            traj_path = raw_dir / traj_path.name
        traj_rows = read_csv_dicts(traj_path)
        # Keep the raw samples in memory here because MBAR reconstruction works
        # directly from per-window coordinates.
        samples = [
            {
                "step": int(float(sample["step"])),
                "x": float(sample["x"]),
                "y": float(sample["y"]),
            }
            for sample in traj_rows
        ]
        windows.append(
            {
                "window_id": int(row["window_id"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "k": float(row["k"]),
                "n_steps": int(float(row["n_steps"])),
                "samples": samples,
            }
        )
    return windows


def load_aus_windows(raw_dir: Path) -> list[dict]:
    rows = read_csv_dicts(raw_dir / "aus_windows.csv")
    windows = []
    for row in sorted(rows, key=lambda item: int(item["window_id"])):
        window_dir = raw_dir / row["window_dir"]
        traj_path = window_dir / row["traj_file"]
        traj_rows = read_csv_dicts(traj_path)
        samples = [
            {
                "step": int(float(sample["step"])),
                "x": float(sample["x"]),
                "y": float(sample["y"]),
            }
            for sample in traj_rows
        ]
        windows.append(
            {
                "window_id": int(row["window_id"]),
                "side": row["side"],
                "depth": int(row["depth"]),
                "iteration": int(float(row.get("iteration", row["depth"]))),
                "parent_window_id": parse_float_or_none(row.get("parent_window_id", "")),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "k": float(row["k"]),
                "n_steps": int(float(row["n_steps"])),
                "n_frames": int(float(row["n_frames"])),
                "cumulative_start": int(float(row["cumulative_start"])),
                "cumulative_end": int(float(row["cumulative_end"])),
                "iteration_budget_start": int(float(row.get("iteration_budget_start", row["cumulative_start"]))),
                "iteration_budget_end": int(float(row.get("iteration_budget_end", row["cumulative_end"]))),
                "iteration_window_count": int(float(row.get("iteration_window_count", 1))),
                "pair_overlap": parse_float_or_none(row.get("pair_overlap", "")),
                "mean_parent": parse_float_or_none(row.get("mean_parent", "")),
                "sigma_parent": parse_float_or_none(row.get("sigma_parent", "")),
                "median_parent_x": parse_float_or_none(row.get("median_parent_x", "")),
                "q_next_x": parse_float_or_none(row.get("q_next_x", "")),
                "target_mean_x": parse_float_or_none(row.get("target_mean_x", "")),
                "local_curvature": parse_float_or_none(row.get("local_curvature", "")),
                "local_slope": parse_float_or_none(row.get("local_slope", "")),
                "derived_k": parse_float_or_none(row.get("derived_k", "")),
                "sample_q_lo": parse_float_or_none(row.get("sample_q_lo", "")),
                "sample_q_hi": parse_float_or_none(row.get("sample_q_hi", "")),
                "target_in_sample_band": row.get("target_in_sample_band", ""),
                "k_clamped_to": row.get("k_clamped_to", ""),
                "alpha": parse_float_or_none(row.get("alpha", "")),
                "fit_method": row.get("fit_method", ""),
                "q_next": parse_float_or_none(row.get("q_next", "")),
                "k_min": parse_float_or_none(row.get("k_min", "")),
                "k_max": parse_float_or_none(row.get("k_max", "")),
                "seed": int(float(row["seed"])),
                "window_dir": row["window_dir"],
                "traj_file": row["traj_file"],
                "samples": samples,
            }
        )
    return windows


def select_tail_samples(
    samples: list[dict],
    step_limit: int,
    keep_fraction: float,
) -> list[float]:
    """Keep only the final fraction of a truncated adaptive-window trajectory."""
    selected = [
        float(sample["x"])
        for sample in samples
        if int(sample["step"]) <= int(step_limit)
    ]
    if not selected:
        return []
    keep_fraction = float(keep_fraction)
    if keep_fraction >= 1.0:
        return selected
    if keep_fraction <= 0.0:
        return selected[-1:]
    discard_count = int(math.floor(len(selected) * (1.0 - keep_fraction)))
    if discard_count >= len(selected):
        discard_count = len(selected) - 1
    return selected[discard_count:]


def build_us_seed_snapshots(raw_dir: Path, ctx: dict, combo: dict, xs: list[float]) -> list[tuple[float, list[float | None], list[float | None]]]:
    windows = load_us_windows(raw_dir)
    if not windows:
        return []
    n_windows = len(windows)
    time_values = [int(value) for value in ctx["time_grid"]["values"]]
    sample_arrays: list[np.ndarray] = [np.zeros(0, dtype=float) for _ in windows]
    reduced_free: np.ndarray | None = None
    snapshots: list[tuple[float, list[float | None], list[float | None]]] = []
    centers = [window["center_x"] for window in windows]
    ks = [window["k"] for window in windows]

    # For a total budget T, US uses the first T / N steps from each of the N
    # umbrella windows.
    for total_steps in time_values:
        steps_per_window = int(total_steps // n_windows)
        for idx, window in enumerate(windows):
            xs_window = [
                sample["x"]
                for sample in window["samples"]
                if sample["step"] <= steps_per_window
            ]
            sample_arrays[idx] = np.asarray(xs_window, dtype=float)
        pmf, reduced_free = us_mbar_snapshot(sample_arrays, centers, ks, xs, ctx, reduced_free)
        snapshots.append((float(total_steps), pmf, [None] * len(xs)))
    return snapshots


def build_aus_seed_snapshots(raw_dir: Path, ctx: dict, combo: dict, xs: list[float]) -> list[tuple[float, list[float | None], list[float | None]]]:
    windows = load_aus_windows(raw_dir)
    if not windows:
        return []
    time_values = [int(value) for value in ctx["time_grid"]["values"]]
    keep_fraction = float(ctx["aus_screen"]["fixed"].get("analysis_tail_fraction", 1.0))
    reduced_free: np.ndarray | None = None
    snapshots: list[tuple[float, list[float | None], list[float | None]]] = []

    # aUS records every budget-sharing stage as one iteration group. Before the
    # frontiers meet that is a left/right child pair; after the match it can be
    # a one-window rescue round or a redistribution round over the allocated
    # umbrella set. Each group's budget is reconstructed from its recorded
    # `iteration_window_count`.
    iterations: dict[int, list[dict]] = {}
    for window in windows:
        iterations.setdefault(int(window["iteration"]), []).append(window)

    for total_steps in time_values:
        sample_arrays: list[np.ndarray] = []
        centers: list[float] = []
        ks: list[float] = []
        for iteration in sorted(iterations):
            group = iterations[iteration]
            iter_start = min(int(window["iteration_budget_start"]) for window in group)
            iter_window_count = max(int(window["iteration_window_count"]) for window in group)
            if total_steps <= iter_start:
                continue
            remaining_total = total_steps - iter_start
            per_window_steps = int(remaining_total // max(iter_window_count, 1))
            for window in group:
                xs_window = select_tail_samples(window["samples"], per_window_steps, keep_fraction)
                if not xs_window:
                    continue
                sample_arrays.append(np.asarray(xs_window, dtype=float))
                centers.append(window["center_x"])
                ks.append(window["k"])
        pmf, reduced_free = us_mbar_snapshot(sample_arrays, centers, ks, xs, ctx, reduced_free)
        snapshots.append((float(total_steps), pmf, [None] * len(xs)))
    return snapshots


def build_mtd_seed_snapshots(
    seed_raw_dir: Path,
    ctx: dict,
    combo: dict,
    xs: list[float],
) -> list[tuple[float, list[float | None], list[float | None]]]:
    left_hills = read_hills(seed_raw_dir / "left" / "meta_hills.csv")
    right_hills = read_hills(seed_raw_dir / "right" / "meta_hills.csv")
    scale = -float(combo["biasfactor"]) / (float(combo["biasfactor"]) - 1.0)
    bias = np.zeros(len(xs), dtype=float)
    xs_np = np.asarray(xs, dtype=float)
    snapshots: list[tuple[float, list[float | None], list[float | None]]] = []
    left_idx = 0
    right_idx = 0
    time_values = [int(value) for value in ctx["time_grid"]["values"]]
    # Two walkers share the total budget equally, so each snapshot uses the
    # first T / 2 steps from both hill streams.
    for total_steps in time_values:
        per_walker_steps = int(total_steps // 2)
        while left_idx < len(left_hills) and left_hills[left_idx]["step"] <= per_walker_steps:
            add_hill_to_bias(bias, xs_np, left_hills[left_idx])
            left_idx += 1
        while right_idx < len(right_hills) and right_hills[right_idx]["step"] <= per_walker_steps:
            add_hill_to_bias(bias, xs_np, right_hills[right_idx])
            right_idx += 1
        pmf = shift_to_zero([scale * value for value in bias.tolist()])
        snapshots.append((float(total_steps), pmf, [None] * len(xs)))
    return snapshots


def direction_pmf_from_raw_segment(
    rep_dir: Path,
    prefix: str,
    ctx: dict,
    base_k: float,
    mid_scale: float,
    xs: list[float],
    left_x: float,
    right_x: float,
) -> list[float | None]:
    # Reconstruct one direction of a switching segment directly from the saved
    # trajectory work values and coordinate histograms.
    trajectories = [read_trajectory(path) for path in list_traj_files(rep_dir, prefix)]
    if not trajectories:
        return [None] * len(xs)
    lambdas = [row["lambda"] for row in trajectories[0]]
    n_time = len(lambdas)
    log_sum_w = np.full(n_time, -np.inf, dtype=float)
    log_sum_hist = np.full((n_time, len(xs)), -np.inf, dtype=float)
    beta = 1.0 / float(ctx["thermal_kT"])
    dx = float(ctx["grid"]["dx"])
    for traj in trajectories:
        for time_idx, row in enumerate(traj):
            log_weight = -beta * row["work"]
            log_sum_w[time_idx] = np.logaddexp(log_sum_w[time_idx], log_weight)
            x_idx = grid_index(xs, dx, row["x"])
            if x_idx >= 0:
                log_sum_hist[time_idx, x_idx] = np.logaddexp(log_sum_hist[time_idx, x_idx], log_weight)
    return hs_reconstruct_segment_from_logs(
        log_sum_w,
        log_sum_hist,
        lambdas,
        xs,
        ctx,
        len(trajectories),
        base_k,
        mid_scale,
        left_x,
        right_x,
    )


def direction_pmf_from_raw(rep_dir: Path, prefix: str, ctx: dict, base_k: float, mid_scale: float, xs: list[float]) -> list[float | None]:
    return direction_pmf_from_raw_segment(
        rep_dir,
        prefix,
        ctx,
        base_k,
        mid_scale,
        xs,
        float(ctx["potential"]["x0"]),
        float(ctx["potential"]["x1"]),
    )


def build_nes_seed_snapshot(raw_dir: Path, ctx: dict, combo: dict, xs: list[float], total_steps: int) -> list[tuple[float, list[float | None], list[float | None]]]:
    base_k = float(combo["k"])
    mid_scale = float(ctx["nes_screen"]["fixed"]["k_midscale"])
    # Bidirectional NES averages forward and backward densities after separate
    # Hummer-Szabo reconstructions.
    fwd = direction_pmf_from_raw(raw_dir, "neq_fwd", ctx, base_k, mid_scale, xs)
    bwd = direction_pmf_from_raw(raw_dir, "neq_bwd", ctx, base_k, mid_scale, xs)
    pmf = average_densities_to_pmf([fwd, bwd], ctx)
    return [(float(total_steps), pmf, [None] * len(xs))]


def load_mines_milestones(raw_dir: Path) -> list[dict]:
    rows = read_csv_dicts(raw_dir / "milestones.csv")
    milestones = []
    for row in sorted(rows, key=lambda item: int(item["milestone_id"])):
        milestones.append(
            {
                "milestone_id": int(row["milestone_id"]),
                "side": row["side"],
                "depth": int(row["depth"]),
                "center_x": float(row["center_x"]),
                "k_eq": float(row["k_eq"]),
                "eq_steps": int(float(row["eq_steps"])),
                "eq_nout": int(float(row["eq_nout"])),
                "cumulative_start": int(float(row["cumulative_start"])),
                "cumulative_end": int(float(row["cumulative_end"])),
                "seed": int(float(row["seed"])),
                "eq_file": row["eq_file"],
                "milestone_dir": row["milestone_dir"],
            }
        )
    return milestones


def load_mines_edges(raw_dir: Path) -> list[dict]:
    rows = read_csv_dicts(raw_dir / "edges.csv")
    edges = []
    for row in sorted(rows, key=lambda item: int(item["edge_id"])):
        edges.append(
            {
                "edge_id": int(row["edge_id"]),
                "type": row["type"],
                "depth": int(row["depth"]),
                "left_id": int(row["left_id"]),
                "right_id": int(row["right_id"]),
                "left_x": float(row["left_x"]),
                "right_x": float(row["right_x"]),
                "k_pull": float(row["k_pull"]),
                "t_neq": int(float(row["t_neq"])),
                "n_traj_per_direction": int(float(row["n_traj_per_direction"])),
                "cumulative_start": int(float(row["cumulative_start"])),
                "cumulative_end": int(float(row["cumulative_end"])),
                "eq_overlap": parse_float_or_none(row["eq_overlap"]),
                "left_reach": parse_float_or_none(row["left_reach"]),
                "right_reach": parse_float_or_none(row["right_reach"]),
                "work_overlap": parse_float_or_none(row["work_overlap"]),
                "resolved": row["resolved"].lower() == "true",
                "edge_dir": row["edge_dir"],
            }
        )
    return edges


def final_works_from_dir(edge_dir: Path, prefix: str) -> list[float]:
    works: list[float] = []
    for path in list_traj_files(edge_dir, prefix):
        rows = read_csv_dicts(path)
        if rows:
            works.append(float(rows[-1]["work"]))
    return works


def align_segment_to_scaffold(
    pmf: list[float | None],
    xs: list[float],
    left_x: float,
    right_x: float,
    scaffold_left: float,
    scaffold_right: float,
) -> list[float | None]:
    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    left_idx = grid_index(xs, dx, left_x)
    right_idx = grid_index(xs, dx, right_x)
    shifts: list[float] = []
    if left_idx >= 0:
        left_val = pmf[left_idx]
        if left_val is not None and math.isfinite(left_val):
            shifts.append(scaffold_left - left_val)
    if right_idx >= 0:
        right_val = pmf[right_idx]
        if right_val is not None and math.isfinite(right_val):
            shifts.append(scaffold_right - right_val)
    if not shifts:
        return pmf
    shift = mean(shifts)
    return [
        None if value is None or not math.isfinite(value) else value + shift
        for value in pmf
    ]


def build_mines_seed_snapshots(raw_dir: Path, ctx: dict, combo: dict, xs: list[float]) -> tuple[list[tuple[float, list[float | None], list[float | None]]], dict]:
    milestones = load_mines_milestones(raw_dir)
    edges = load_mines_edges(raw_dir)
    if not milestones and not edges:
        return [], {}

    beta = 1.0 / float(ctx["thermal_kT"])
    k_pull = float(combo["k_pull"])
    mid_scale = float(ctx["nes_screen"]["fixed"].get("k_midscale", 1.0))
    edge_cache: dict[int, dict] = {}
    reduced_edges: list[dict] = []

    # First reduce every edge independently into local PMFs and BAR offsets.
    for edge in edges:
        edge_dir = raw_dir / edge["edge_dir"]
        forward_accumulators = None
        backward_accumulators = None
        if edge["type"] == "parent":
            fwd_acc_path = edge_dir / "forward_accumulators.json"
            bwd_acc_path = edge_dir / "backward_accumulators.json"
            if fwd_acc_path.exists():
                forward_accumulators = load_json(fwd_acc_path)
            if bwd_acc_path.exists():
                backward_accumulators = load_json(bwd_acc_path)
        fwd_works = final_works_from_dir(edge_dir, "neq_fwd")
        bwd_works = final_works_from_dir(edge_dir, "neq_bwd")
        fwd_pmf = direction_pmf_from_raw_segment(
            edge_dir,
            "neq_fwd",
            ctx,
            k_pull,
            mid_scale,
            xs,
            edge["left_x"],
            edge["right_x"],
        )
        bwd_pmf = direction_pmf_from_raw_segment(
            edge_dir,
            "neq_bwd",
            ctx,
            k_pull,
            mid_scale,
            xs,
            edge["left_x"],
            edge["right_x"],
        )
        local_pmf = average_densities_to_pmf([fwd_pmf, bwd_pmf], ctx)
        delta_f, bar_sigma = bar_delta_f(fwd_works, bwd_works, beta)
        edge_cache[edge["edge_id"]] = {
            "edge": edge,
            "local_pmf": local_pmf,
            "delta_f": delta_f,
            "bar_sigma": bar_sigma,
            "forward_works": fwd_works,
            "backward_works": bwd_works,
        }
        reduced_edges.append(
            {
                "edge_id": edge["edge_id"],
                "type": edge["type"],
                "left_x": edge["left_x"],
                "right_x": edge["right_x"],
                "delta_f": delta_f,
                "bar_sigma": bar_sigma,
                "work_overlap": edge["work_overlap"],
                "cumulative_end": edge["cumulative_end"],
                "n_forward": len(fwd_works),
                "n_backward": len(bwd_works),
                "forward_works": fwd_works,
                "backward_works": bwd_works,
                "forward_accumulators": forward_accumulators,
                "backward_accumulators": backward_accumulators,
            }
        )

    snapshots: list[tuple[float, list[float | None], list[float | None]]] = []
    # Then expose those reduced objects on the benchmark time grid by only
    # using edges whose cumulative cost fits within each budget.
    for total_steps in [int(value) for value in ctx["time_grid"]["values"]]:
        available_adjacent = [
            edge_cache[edge["edge_id"]]
            for edge in edges
            if edge["type"] == "adjacent" and edge["cumulative_end"] <= total_steps
        ]
        if available_adjacent:
            available_adjacent.sort(key=lambda item: item["edge"]["left_x"])
            scaffold: dict[float, float] = {available_adjacent[0]["edge"]["left_x"]: 0.0}
            aligned_pmfs: list[list[float | None]] = []
            for item in available_adjacent:
                left_x = item["edge"]["left_x"]
                right_x = item["edge"]["right_x"]
                if left_x not in scaffold:
                    scaffold[left_x] = 0.0
                scaffold[right_x] = scaffold[left_x] + item["delta_f"]
                aligned_pmfs.append(
                    align_segment_to_scaffold(
                        item["local_pmf"],
                        xs,
                        left_x,
                        right_x,
                        scaffold[left_x],
                        scaffold[right_x],
                    )
                )
            pmf = combine_absolute_pmfs(aligned_pmfs, ctx)
            snapshots.append((float(total_steps), pmf, [None] * len(xs)))
            continue

        parent_candidates = [
            edge_cache[edge["edge_id"]]
            for edge in edges
            if edge["type"] == "parent" and edge["cumulative_end"] <= total_steps
        ]
        if parent_candidates:
            parent_candidates.sort(key=lambda item: item["edge"]["cumulative_end"])
            pmf = parent_candidates[-1]["local_pmf"]
            snapshots.append((float(total_steps), pmf, [None] * len(xs)))
            continue

        snapshots.append((float(total_steps), [None] * len(xs), [None] * len(xs)))

    reduced_payload = {
        "milestones": milestones,
        "edges": reduced_edges,
    }
    return snapshots, reduced_payload


def generate_us_gif(raw_dir: Path, out_path: Path, ctx: dict, combo: dict, seed: int) -> None:
    windows = load_us_windows(raw_dir)
    if not windows:
        return
    xmin = float(ctx["grid"]["xmin"])
    xmax = float(ctx["grid"]["xmax"])
    n_bins = 60
    bins = np.linspace(xmin, xmax, n_bins + 1)
    frame_limits = np.linspace(
        0,
        max(max(sample["step"] for sample in window["samples"]) for window in windows),
        GIF_FRAMES,
    )
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(windows)))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    def animate(frame_idx: int) -> None:
        limit = frame_limits[frame_idx]
        ax.clear()
        for color, window in zip(colors, windows):
            values = [sample["x"] for sample in window["samples"] if sample["step"] <= limit]
            if not values:
                continue
            hist, edges = np.histogram(values, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, hist, color=color, lw=1.4, alpha=0.9, label=f"w{window['window_id']}")
            ax.axvline(window["center_x"], color=color, linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        add_annotation(
            ax,
            "US seed 101 evolution",
            [
                f"k0={ctx['potential']['k0']}, x0={ctx['potential']['x0']}, k1={ctx['potential']['k1']}, x1={ctx['potential']['x1']}, E1={ctx['potential']['E1']}",
                f"US combo: {combo['label']} (k={combo['k']}, dx={combo['dx']})",
                f"seed={seed}, cumulative time per window <= {int(limit)} steps",
                "dashed lines: restraint centers",
            ],
        )
        if len(windows) <= 12:
            ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    save_gif(fig, animate, GIF_FRAMES, out_path)


def generate_mtd_gif(seed_raw_dir: Path, out_path: Path, ctx: dict, combo: dict, seed: int) -> None:
    left_rows = read_csv_dicts(seed_raw_dir / "left" / "meta_traj.csv")
    right_rows = read_csv_dicts(seed_raw_dir / "right" / "meta_traj.csv")
    left = [{"step": int(float(row["step"])), "x": float(row["x"])} for row in left_rows]
    right = [{"step": int(float(row["step"])), "x": float(row["x"])} for row in right_rows]
    max_step = max(left[-1]["step"], right[-1]["step"])
    frame_limits = np.linspace(0, max_step, GIF_FRAMES)
    xmin = float(ctx["grid"]["xmin"])
    xmax = float(ctx["grid"]["xmax"])
    bins = np.linspace(xmin, xmax, 61)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    def animate(frame_idx: int) -> None:
        limit = frame_limits[frame_idx]
        ax.clear()
        left_values = [row["x"] for row in left if row["step"] <= limit]
        right_values = [row["x"] for row in right if row["step"] <= limit]
        both_values = left_values + right_values
        for values, color, label in (
            (left_values, "#0b6e4f", "left walker"),
            (right_values, "#c84c09", "right walker"),
            (both_values, "#2f4858", "combined"),
        ):
            if not values:
                continue
            hist, edges = np.histogram(values, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, hist, color=color, lw=2.0, label=label)
        y_top = max(ax.get_ylim()[1], 1.0e-6)
        if left_values:
            ax.scatter([left_values[-1]], [0.96 * y_top], color="#0b6e4f", marker="v", s=55, zorder=5, label="left x(t)")
        if right_values:
            ax.scatter([right_values[-1]], [0.90 * y_top], color="#c84c09", marker="v", s=55, zorder=5, label="right x(t)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        add_annotation(
            ax,
            "WT-MTD seed 101 evolution",
            [
                f"k0={ctx['potential']['k0']}, x0={ctx['potential']['x0']}, k1={ctx['potential']['k1']}, x1={ctx['potential']['x1']}, E1={ctx['potential']['E1']}",
                f"w0={combo['w0']}, sigma={combo['sigma']}, biasfactor={combo['biasfactor']}, stride={combo['stride']}",
                f"seed={seed}, cumulative total time={int(2 * limit)} steps",
                "triangles: instantaneous walker positions",
            ],
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    save_gif(fig, animate, GIF_FRAMES, out_path)


def generate_nes_gif(raw_dir: Path, out_path: Path, ctx: dict, combo: dict, seed: int, total_steps: int) -> None:
    fwd_trajs = [read_trajectory(path) for path in list_traj_files(raw_dir, "neq_fwd")]
    bwd_trajs = [read_trajectory(path) for path in list_traj_files(raw_dir, "neq_bwd")]
    if not fwd_trajs or not bwd_trajs:
        return
    n_points = min(len(fwd_trajs[0]), len(bwd_trajs[0]))
    frame_indices = np.linspace(0, n_points - 1, GIF_FRAMES, dtype=int)
    xmin = float(ctx["grid"]["xmin"])
    xmax = float(ctx["grid"]["xmax"])
    bins = np.linspace(xmin, xmax, 61)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    def animate(frame_idx: int) -> None:
        sample_idx = int(frame_indices[frame_idx])
        ax.clear()
        fwd_rows = [traj[min(sample_idx, len(traj) - 1)] for traj in fwd_trajs]
        bwd_rows = [traj[min(sample_idx, len(traj) - 1)] for traj in bwd_trajs]
        fwd_values = [row["x"] for row in fwd_rows]
        bwd_values = [row["x"] for row in bwd_rows]
        for values, color, label in (
            (fwd_values, "#0b6e4f", "forward shootings"),
            (bwd_values, "#c84c09", "backward shootings"),
        ):
            if not values:
                continue
            hist, edges = np.histogram(values, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, hist, color=color, lw=2.0, label=label)
        if fwd_rows:
            ax.axvline(center_from_lambda(ctx, fwd_rows[0]["lambda"]), color="#0b6e4f", linestyle="--", linewidth=1.2, alpha=0.75)
        if bwd_rows:
            ax.axvline(center_from_lambda(ctx, bwd_rows[0]["lambda"]), color="#c84c09", linestyle="--", linewidth=1.2, alpha=0.75)
        progress = 0.0 if n_points <= 1 else float(sample_idx) / float(n_points - 1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("x")
        ax.set_ylabel("Density")
        add_annotation(
            ax,
            "NES seed 101 switching evolution",
            [
                f"k0={ctx['potential']['k0']}, x0={ctx['potential']['x0']}, k1={ctx['potential']['k1']}, x1={ctx['potential']['x1']}, E1={ctx['potential']['E1']}",
                f"NES combo: {combo['label']} (k={combo['k']}, k_midscale={ctx['nes_screen']['fixed']['k_midscale']})",
                f"seed={seed}, cumulative total time≈{int(round(progress * total_steps))} steps",
                "dashed lines: current restraint centers",
            ],
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    save_gif(fig, animate, GIF_FRAMES, out_path)


def process_us_seed(system_root: Path, ctx: dict, combo_label: str, seed: int, make_gif: bool) -> None:
    xs = build_grid(ctx["grid"]["xmin"], ctx["grid"]["xmax"], ctx["grid"]["dx"])
    combo = next(combo for combo in ctx["us_screen"]["combos"] if combo["label"] == combo_label)
    raw_dir = system_root / "US" / combo_label / "raw" / f"seed_{seed}"
    processed_path = system_root / "US" / combo_label / "processed" / f"seed_{seed}.dat"
    reduced_path = system_root / "US" / combo_label / "reduced" / f"seed_{seed}.csv"
    gif_path = system_root / "US" / combo_label / "gifs" / f"seed_{seed}.gif"

    snapshots = build_us_seed_snapshots(raw_dir, ctx, combo, xs)
    write_method_dat(processed_path, xs, snapshots)

    windows = load_us_windows(raw_dir)
    # Retain a compact cross-window sample file before deleting the bulky raw
    # per-window trajectories.
    reduced_rows = []
    for window in windows:
        for sample in window["samples"]:
            reduced_rows.append(
                {
                    "window_id": window["window_id"],
                    "center_x": f"{window['center_x']:.10f}",
                    "k": f"{window['k']:.10f}",
                    "step": sample["step"],
                    "x": f"{sample['x']:.10f}",
                    "y": f"{sample['y']:.10f}",
                }
            )
    write_reduced_csv(reduced_path, downsample_evenly(reduced_rows))
    if make_gif:
        generate_us_gif(raw_dir, gif_path, ctx, combo, seed)
    shutil.rmtree(raw_dir, ignore_errors=True)


def process_aus_seed(system_root: Path, ctx: dict, combo_label: str, seed: int) -> None:
    xs = build_grid(ctx["grid"]["xmin"], ctx["grid"]["xmax"], ctx["grid"]["dx"])
    combo = next(combo for combo in ctx["aus_screen"]["combos"] if combo["label"] == combo_label)
    raw_dir = system_root / "AUS" / combo_label / "raw" / f"seed_{seed}"
    processed_path = system_root / "AUS" / combo_label / "processed" / f"seed_{seed}.dat"
    reduced_path = system_root / "AUS" / combo_label / "reduced" / f"seed_{seed}.csv"

    snapshots = build_aus_seed_snapshots(raw_dir, ctx, combo, xs)
    write_method_dat(processed_path, xs, snapshots)

    windows = load_aus_windows(raw_dir)
    reduced_rows = []
    for window in windows:
        for sample in window["samples"]:
            reduced_rows.append(
                {
                    "window_id": window["window_id"],
                    "side": window["side"],
                    "depth": window["depth"],
                    "iteration": window["iteration"],
                    "parent_window_id": "" if window["parent_window_id"] is None else int(window["parent_window_id"]),
                    "center_x": f"{window['center_x']:.10f}",
                    "k": f"{window['k']:.10f}",
                    "mean_parent": "" if window["mean_parent"] is None else f"{window['mean_parent']:.10f}",
                    "sigma_parent": "" if window["sigma_parent"] is None else f"{window['sigma_parent']:.10f}",
                    "median_parent_x": "" if window["median_parent_x"] is None else f"{window['median_parent_x']:.10f}",
                    "q_next_x": "" if window["q_next_x"] is None else f"{window['q_next_x']:.10f}",
                    "target_mean_x": "" if window["target_mean_x"] is None else f"{window['target_mean_x']:.10f}",
                    "local_curvature": "" if window["local_curvature"] is None else f"{window['local_curvature']:.10f}",
                    "local_slope": "" if window["local_slope"] is None else f"{window['local_slope']:.10f}",
                    "derived_k": "" if window["derived_k"] is None else f"{window['derived_k']:.10f}",
                    "sample_q_lo": "" if window["sample_q_lo"] is None else f"{window['sample_q_lo']:.10f}",
                    "sample_q_hi": "" if window["sample_q_hi"] is None else f"{window['sample_q_hi']:.10f}",
                    "target_in_sample_band": window.get("target_in_sample_band", ""),
                    "k_clamped_to": window["k_clamped_to"],
                    "alpha": "" if window["alpha"] is None else f"{window['alpha']:.10f}",
                    "fit_method": window.get("fit_method", ""),
                    "q_next": "" if window["q_next"] is None else f"{window['q_next']:.10f}",
                    "k_min": "" if window["k_min"] is None else f"{window['k_min']:.10f}",
                    "k_max": "" if window["k_max"] is None else f"{window['k_max']:.10f}",
                    "step": sample["step"],
                    "x": f"{sample['x']:.10f}",
                    "y": f"{sample['y']:.10f}",
                    "cumulative_start": window["cumulative_start"],
                    "cumulative_end": window["cumulative_end"],
                    "iteration_budget_start": window["iteration_budget_start"],
                    "iteration_budget_end": window["iteration_budget_end"],
                    "iteration_window_count": window["iteration_window_count"],
                    "pair_overlap": "" if window["pair_overlap"] is None else f"{window['pair_overlap']:.10f}",
                }
            )
    write_reduced_csv(reduced_path, reduced_rows)
    shutil.rmtree(raw_dir, ignore_errors=True)


def process_mtd_seed(system_root: Path, ctx: dict, combo_label: str, seed: int, make_gif: bool) -> None:
    xs = build_grid(ctx["grid"]["xmin"], ctx["grid"]["xmax"], ctx["grid"]["dx"])
    combo = next(combo for combo in ctx["mtd_screen"]["combos"] if combo["label"] == combo_label)
    raw_dir = system_root / "MTD" / combo_label / "raw" / f"seed_{seed}"
    processed_path = system_root / "MTD" / combo_label / "processed" / f"seed_{seed}.dat"
    reduced_path = system_root / "MTD" / combo_label / "reduced" / f"seed_{seed}.csv"
    gif_path = system_root / "MTD" / combo_label / "gifs" / f"seed_{seed}.gif"

    snapshots = build_mtd_seed_snapshots(raw_dir, ctx, combo, xs)
    write_method_dat(processed_path, xs, snapshots)

    # Keep one evenly downsampled combined walker trajectory for inspection and
    # GIF generation, then drop the raw walker directories.
    reduced_rows = []
    for walker in ("left", "right"):
        for row in read_csv_dicts(raw_dir / walker / "meta_traj.csv"):
            reduced_rows.append(
                {
                    "walker": walker,
                    "step": int(float(row["step"])),
                    "x": f"{float(row['x']):.10f}",
                    "y": f"{float(row['y']):.10f}",
                    "base_u": f"{float(row['base_u']):.10f}",
                    "meta_u": f"{float(row['meta_u']):.10f}",
                    "total_u": f"{float(row['total_u']):.10f}",
                }
            )
    write_reduced_csv(reduced_path, downsample_evenly(reduced_rows))
    if make_gif:
        generate_mtd_gif(raw_dir, gif_path, ctx, combo, seed)
    shutil.rmtree(raw_dir, ignore_errors=True)


def process_nes_seed_time(
    system_root: Path,
    ctx: dict,
    combo_label: str,
    seed: int,
    total_steps: int,
    make_gif: bool,
    retain_reduced: bool,
) -> None:
    xs = build_grid(ctx["grid"]["xmin"], ctx["grid"]["xmax"], ctx["grid"]["dx"])
    combo = next(combo for combo in ctx["nes_screen"]["combos"] if combo["label"] == combo_label)
    label = time_label(total_steps)
    raw_dir = system_root / "NES" / combo_label / "raw" / label / f"seed_{seed}"
    processed_path = system_root / "NES" / combo_label / "processed" / f"seed_{seed}.dat"
    reduced_path = system_root / "NES" / combo_label / "reduced" / f"seed_{seed}.csv"
    gif_path = system_root / "NES" / combo_label / "gifs" / f"{label}.gif"

    snapshot = build_nes_seed_snapshot(raw_dir, ctx, combo, xs, total_steps)
    overwrite = not processed_path.exists()
    write_method_dat(processed_path, xs, snapshot, overwrite=overwrite)

    # Only the longest switching-time case keeps a reduced trajectory snapshot;
    # shorter budgets are summarized immediately and then discarded.
    if retain_reduced:
        rows = []
        for direction, prefix in (("forward", "neq_fwd"), ("backward", "neq_bwd")):
            for traj_path in list_traj_files(raw_dir, prefix):
                traj_idx = int(traj_path.stem.split("_")[-1])
                for row in read_csv_dicts(traj_path):
                    rows.append(
                        {
                            "direction": direction,
                            "traj_idx": traj_idx,
                            "step": int(float(row["step"])),
                            "lambda": f"{float(row['lambda']):.10f}",
                            "x": f"{float(row['x']):.10f}",
                            "y": f"{float(row['y']):.10f}",
                            "work": f"{float(row['work']):.10f}",
                        }
                    )
        write_reduced_csv(reduced_path, downsample_evenly(rows))
    if make_gif:
        generate_nes_gif(raw_dir, gif_path, ctx, combo, seed, total_steps)
    shutil.rmtree(raw_dir, ignore_errors=True)


def process_mines_seed(system_root: Path, ctx: dict, combo_label: str, seed: int) -> None:
    xs = build_grid(ctx["grid"]["xmin"], ctx["grid"]["xmax"], ctx["grid"]["dx"])
    combo = next(combo for combo in ctx["mines_screen"]["combos"] if combo["label"] == combo_label)
    raw_dir = system_root / "MINES" / combo_label / "raw" / f"seed_{seed}"
    processed_path = system_root / "MINES" / combo_label / "processed" / f"seed_{seed}.dat"
    reduced_path = system_root / "MINES" / combo_label / "reduced" / f"seed_{seed}.json"

    snapshots, reduced_payload = build_mines_seed_snapshots(raw_dir, ctx, combo, xs)
    write_method_dat(processed_path, xs, snapshots)
    write_json(reduced_path, reduced_payload)
    shutil.rmtree(raw_dir, ignore_errors=True)


def write_ranking_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_seed_replicates(seed_paths: Iterable[Path]) -> list[list[tuple[float, list[float | None], list[float | None]]]]:
    replicates = []
    for path in seed_paths:
        if path.exists():
            replicates.append(read_method_dat(path))
    return replicates


def combo_row(base: dict, combo: dict, selected: bool) -> dict:
    row = dict(base)
    row["selected"] = "true" if selected else "false"
    for key in ("label", "direction", "fit_method", "k", "alpha", "q_next", "k_min", "k_max", "dx", "n_windows", "steps_per_window", "total_steps", "k_midscale", "k_pull", "biasfactor"):
        if key in combo:
            row[key] = combo[key]
    return row


def write_summary(
    path: Path,
    system_root: Path,
    ctx: dict,
    selection: dict,
    us_rankings: list[dict],
    aus_rankings: list[dict],
    nes_rankings: list[dict],
    mines_rankings: list[dict],
    mtd_rankings: list[dict],
) -> None:
    lines = [
        "1D DoubleWell benchmark summary",
        "",
        "System construction",
        "name: double-well",
        f"k0: {ctx['potential']['k0']}",
        f"x0: {ctx['potential']['x0']}",
        f"k1: {ctx['potential']['k1']}",
        f"x1: {ctx['potential']['x1']}",
        f"E1: {ctx['potential']['E1']}",
        "",
        "Shared dynamics settings",
        f"thermal_kT: {ctx['thermal_kT']}",
        f"dt: {ctx['dt']}",
        f"gamma: {ctx['gamma']}",
        f"PMF grid: x in [{ctx['grid']['xmin']}, {ctx['grid']['xmax']}] with dx={ctx['grid']['dx']}",
        f"RMSE evaluation grid: x in [{ctx['rmse_eval_grid']['xmin']}, {ctx['rmse_eval_grid']['xmax']}] with dx={ctx['rmse_eval_grid']['dx']}",
        f"replicate seeds: {', '.join(str(seed) for seed in ctx['seeds'])}",
        f"time grid (step units): {', '.join(str(int(v)) for v in ctx['time_grid']['values'])}",
        "",
        "US screen",
        f"k values: {', '.join(str(v) for v in ctx['us_screen']['k_values'])}",
        f"dx values: {', '.join(str(v) for v in ctx['us_screen']['dx_values'])}",
        f"total_steps per combo: {ctx['us_screen']['fixed']['total_steps']}",
        f"saved sample stride: {ctx['us_screen']['fixed']['sample_stride_steps']} steps",
        "",
        "AUS screen",
        f"q_next values: {', '.join(str(v) for v in ctx['aus_screen'].get('q_next_values', []))}",
        f"alpha values: {', '.join(str(v) for v in ctx['aus_screen'].get('alpha_values', []))}",
        f"fit methods: {', '.join(str(v) for v in ctx['aus_screen'].get('fit_method_values', []))}",
        f"k_min values: {', '.join(str(v) for v in ctx['aus_screen'].get('k_min_values', []))}",
        f"k_max values: {', '.join(str(v) for v in ctx['aus_screen'].get('k_max_values', []))}",
        f"frontier grid dx: {ctx['aus_screen']['fixed'].get('grid_dx', ctx['grid']['dx'])}",
        f"start_x_left: {ctx['aus_screen']['fixed'].get('start_x_left', ctx['basins']['left'])}",
        f"start_x_right: {ctx['aus_screen']['fixed'].get('start_x_right', ctx['basins']['right'])}",
        f"endpoint k: {ctx['aus_screen']['fixed'].get('endpoint_k', 1.0)}",
        f"eq_steps per adaptive window: {ctx['aus_screen']['fixed'].get('eq_steps', '')}",
        f"eq_nout per adaptive window: {ctx['aus_screen']['fixed'].get('eq_nout', '')}",
        f"total_steps per combo: {ctx['aus_screen']['fixed'].get('total_steps', int(ctx['time_grid']['values'][-1]))}",
        f"PMF analysis tail fraction per adaptive window: {ctx['aus_screen']['fixed'].get('analysis_tail_fraction', 1.0)}",
        "placement rule: grow one left child and one right child each iteration; the left child uses q_next(parent samples), the right child uses q_{1-next}(parent samples), and each child center is placed from the parent median and alpha",
        "k rule: estimate F' from the entire PMF reconstructed from the parent umbrella only, excluding all ancestor windows; the screened fit method is stored in the combo label and is either the original 4-term polynomial fit or a cubic-spline fit; the left child derives k only when F'(q_next) > 0, the right child derives k only when F'(q_{1-next}) < 0, otherwise the child is placed at the target quantile with k_min",
        "stop rule: stop before launching the next pair when q^m_{next,left} > q^m_{1-next,right}",
        f"post-match refine metric: {ctx['aus_screen']['fixed'].get('refine_metric', 'interested_region_fraction_of_average_ess')}",
        f"post-match curvature-fit method: {ctx['aus_screen']['fixed'].get('refine_fit_method', 'combo_specific')}",
        f"post-match local-fit half-width: {ctx['aus_screen']['fixed'].get('refine_half_width_points', 5)} grid points",
        f"post-match k_addition: {ctx['aus_screen']['fixed'].get('refine_k_addition', 10.0)}",
        f"post-match rescue k_max: {ctx['aus_screen']['fixed'].get('refine_rescue_k_max', 100.0)}",
        f"post-match minimum ESS fraction of the interested-region average before redistribution: {ctx['aus_screen']['fixed'].get('refine_ess_min_fraction', ctx['aus_screen']['fixed'].get('refine_ess_threshold', 0.05))}",
        "post-match budget rule: once the frontiers meet, reconstruct the PMF over the interested region between start_x_left and start_x_right and target the current lowest fractional-ESS resolved bin; the polynomial combo now always adds a target-bin rescue window, estimates curvature from a local 4-term polynomial fit when the target region is resolved and from the first resolved points on each side with a 3-term polynomial fit when it is unresolved, sets k_rescue = |curvature| + 10 at the target bin regardless of curvature sign, doubles that rescue spring if the sampled 1-next to q_next band still misses the target, and marks the grid point unresolvable once that retry hits k_rescue_max, while the cubic combo keeps the earlier extension-first cubic-spline rescue path; after the interested region clears the fractional ESS target, add one more eq_steps block to every allocated umbrella before re-checking ESS again",
        f"max_iterations: {ctx['aus_screen']['fixed'].get('max_iterations', '')}",
        "",
        "NES screen",
        f"k values: {', '.join(str(v) for v in ctx['nes_screen']['k_values'])}",
        f"eq_steps: {ctx['nes_screen']['fixed']['eq_steps']}",
        f"eq_nout: {ctx['nes_screen']['fixed']['eq_nout']}",
        f"n_traj_per_direction: {ctx['nes_screen']['fixed']['n_traj_per_direction']}",
        f"neq_nout per switching trajectory: {ctx['nes_screen']['fixed']['neq_nout']}",
        f"k_midscale: {ctx['nes_screen']['fixed']['k_midscale']}",
        "",
        "MINES screen",
        f"k_pull values: {', '.join(str(v) for v in ctx['mines_screen']['k_pull_values'])}",
        f"frontier grid dx: {ctx['mines_screen']['fixed'].get('grid_dx', ctx['rmse_eval_grid']['dx'])}",
        f"eq_steps: {ctx['mines_screen']['fixed'].get('eq_steps', '')}",
        f"eq_nout: {ctx['mines_screen']['fixed'].get('eq_nout', '')}",
        f"n_traj_per_direction: {ctx['mines_screen']['fixed'].get('n_traj_per_direction', '')}",
        f"t_neq: {ctx['mines_screen']['fixed'].get('t_neq', '')}",
        f"neq_nout: {ctx['mines_screen']['fixed'].get('neq_nout', '')}",
        f"ESS_min: {ctx['mines_screen']['fixed'].get('ess_min', '')}",
        f"overlap_min: {ctx['mines_screen']['fixed'].get('overlap_min', '')}",
        f"work_overlap_min: {ctx['mines_screen']['fixed'].get('work_overlap_min', '')}",
        f"max_depth_per_side: {ctx['mines_screen']['fixed'].get('max_depth_per_side', '')}",
        "",
        "WT-MTD screen",
        f"biasfactor values: {', '.join(str(v) for v in ctx['mtd_screen']['biasfactor_values'])}",
        f"total_steps: {ctx['mtd_screen']['fixed']['total_steps']}",
        f"per_walker_steps: {ctx['mtd_screen']['fixed']['per_walker_steps']}",
        f"saved sample stride: {ctx['mtd_screen']['fixed']['sample_stride_steps']} steps",
        f"w0: {ctx['mtd_screen']['fixed']['w0']}",
        f"sigma: {ctx['mtd_screen']['fixed']['sigma']}",
        f"stride: {ctx['mtd_screen']['fixed']['stride']}",
        "",
        "Selection rule",
        "1. Prefer combinations with at least 95% finite coverage on the central x region.",
        "2. Within the same coverage-success class, prefer higher central-region finite coverage.",
        "3. Break remaining ties using lower final-time RMSE, then lower final mean variance, then lower time-averaged RMSE.",
        "",
        "Selected benchmark-facing files",
        f"us.dat -> {selection['US']['label']}",
        f"aus.dat -> {selection['AUS']['label']}",
        f"nes.dat -> {selection['NES']['label']}",
        f"mines.dat -> {selection['MINES']['label']}",
        f"mtd.dat -> {selection['MTD']['label']}",
        "",
        "Streaming cleanup policy",
        "US and MTD keep only processed PMFs, method metadata, one 1000-sample reduced trajectory file per seed, and selected GIFs.",
        "AUS keeps processed PMFs plus one reduced adaptive-window trajectory file per seed.",
        "NES keeps processed PMFs for every target budget and a 1000-sample reduced trajectory file only for the longest switching-time case.",
        "MINES keeps processed PMFs plus reduced milestone/work summaries rather than all raw switching trajectories.",
        "",
        "Filesystem layout",
        f"system root: {system_root}",
        "US seed PMFs: US/<combo>/processed/seed_<seed>.dat",
        "AUS seed PMFs: AUS/<combo>/processed/seed_<seed>.dat",
        "NES seed PMFs: NES/<combo>/processed/seed_<seed>.dat",
        "MINES seed PMFs: MINES/<combo>/processed/seed_<seed>.dat",
        "MTD seed PMFs: MTD/<combo>/processed/seed_<seed>.dat",
        "benchmark outputs: benchmark/selected/{us,aus,nes,mines,mtd}.dat",
        "benchmark figures: benchmark/figures/",
        "benchmark gifs: benchmark/gifs/",
        "",
        "US ranking rows",
    ]
    for row in us_rankings:
        lines.append(
            f"{row['label']}: rmse={row['final_rmse']:.6f}, coverage={row['central_coverage_fraction']:.3f}, selected={row['selected']}"
        )
    lines.extend(["", "AUS ranking rows"])
    for row in aus_rankings:
        lines.append(
            f"{row['label']}: rmse={row['final_rmse']:.6f}, coverage={row['central_coverage_fraction']:.3f}, selected={row['selected']}"
        )
    lines.extend(["", "NES ranking rows"])
    for row in nes_rankings:
        lines.append(
            f"{row['label']}: rmse={row['final_rmse']:.6f}, coverage={row['central_coverage_fraction']:.3f}, selected={row['selected']}"
        )
    lines.extend(["", "MINES ranking rows"])
    for row in mines_rankings:
        lines.append(
            f"{row['label']}: rmse={row['final_rmse']:.6f}, coverage={row['central_coverage_fraction']:.3f}, selected={row['selected']}"
        )
    lines.extend(["", "MTD ranking rows"])
    for row in mtd_rankings:
        lines.append(
            f"{row['label']}: rmse={row['final_rmse']:.6f}, coverage={row['central_coverage_fraction']:.3f}, selected={row['selected']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(system_root: Path, ctx: dict) -> None:
    xs = build_grid(ctx["grid"]["xmin"], ctx["grid"]["xmax"], ctx["grid"]["dx"])
    eval_xs = build_eval_grid(ctx)
    analytic_eval = analytic_doublewell_profile(eval_xs, ctx)
    selected_root = system_root / "benchmark" / "selected"
    figures_root = system_root / "benchmark" / "figures"
    gifs_root = system_root / "benchmark" / "gifs"
    shutil.rmtree(selected_root, ignore_errors=True)
    shutil.rmtree(figures_root, ignore_errors=True)
    shutil.rmtree(gifs_root, ignore_errors=True)
    selected_root.mkdir(parents=True, exist_ok=True)

    # Rebuild method-level aggregate PMFs from the per-seed processed files, rank
    # the screened combinations, and publish one benchmark-facing winner per
    # method under `benchmark/selected`.
    us_rankings: list[dict] = []
    us_selection: dict | None = None
    for combo in ctx["us_screen"]["combos"]:
        reps = load_seed_replicates(
            system_root / "US" / combo["label"] / "processed" / f"seed_{seed}.dat"
            for seed in ctx["seeds"]
        )
        aggregate = aggregate_replicates(reps)
        write_method_dat(system_root / "US" / combo["label"] / "processed" / "aggregate.dat", xs, aggregate)
        metrics = evaluate_combo(aggregate, xs, eval_xs, analytic_eval)
        metrics["label"] = combo["label"]
        us_rankings.append(combo_row(metrics, combo, False))
    us_rankings.sort(key=selection_sort_key)
    us_selection = us_rankings[0]
    for row in us_rankings:
        row["selected"] = "true" if row["label"] == us_selection["label"] else "false"
    write_ranking_csv(system_root / "US" / "rankings" / "all.csv", us_rankings)
    selected_us = read_method_dat(system_root / "US" / us_selection["label"] / "processed" / "aggregate.dat")
    write_method_dat(selected_root / "us.dat", xs, selected_us)

    aus_rankings: list[dict] = []
    aus_selection: dict | None = None
    for combo in ctx["aus_screen"]["combos"]:
        reps = load_seed_replicates(
            system_root / "AUS" / combo["label"] / "processed" / f"seed_{seed}.dat"
            for seed in ctx["seeds"]
        )
        aggregate = aggregate_replicates(reps)
        write_method_dat(system_root / "AUS" / combo["label"] / "processed" / "aggregate.dat", xs, aggregate)
        metrics = evaluate_combo(aggregate, xs, eval_xs, analytic_eval)
        metrics["label"] = combo["label"]
        aus_rankings.append(combo_row(metrics, combo, False))
    if aus_rankings:
        aus_rankings.sort(key=selection_sort_key)
        aus_selection = aus_rankings[0]
        for row in aus_rankings:
            row["selected"] = "true" if row["label"] == aus_selection["label"] else "false"
        write_ranking_csv(system_root / "AUS" / "rankings" / "all.csv", aus_rankings)
        selected_aus = read_method_dat(system_root / "AUS" / aus_selection["label"] / "processed" / "aggregate.dat")
        write_method_dat(selected_root / "aus.dat", xs, selected_aus)
    else:
        aus_selection = {"label": "missing", "status": "missing"}
        write_method_dat(selected_root / "aus.dat", xs, [])

    nes_rankings: list[dict] = []
    nes_selection: dict | None = None
    for combo in ctx["nes_screen"]["combos"]:
        reps = load_seed_replicates(
            system_root / "NES" / combo["label"] / "processed" / f"seed_{seed}.dat"
            for seed in ctx["seeds"]
        )
        aggregate = aggregate_replicates(reps)
        write_method_dat(system_root / "NES" / combo["label"] / "processed" / "aggregate.dat", xs, aggregate)
        metrics = evaluate_combo(aggregate, xs, eval_xs, analytic_eval)
        metrics["label"] = combo["label"]
        nes_rankings.append(combo_row(metrics, combo, False))
    nes_rankings.sort(key=selection_sort_key)
    nes_selection = nes_rankings[0]
    for row in nes_rankings:
        row["selected"] = "true" if row["label"] == nes_selection["label"] else "false"
    write_ranking_csv(system_root / "NES" / "rankings" / "all.csv", nes_rankings)
    selected_nes = read_method_dat(system_root / "NES" / nes_selection["label"] / "processed" / "aggregate.dat")
    write_method_dat(selected_root / "nes.dat", xs, selected_nes)

    mines_rankings: list[dict] = []
    mines_selection: dict | None = None
    for combo in ctx["mines_screen"]["combos"]:
        reps = load_seed_replicates(
            system_root / "MINES" / combo["label"] / "processed" / f"seed_{seed}.dat"
            for seed in ctx["seeds"]
        )
        aggregate = aggregate_replicates(reps)
        write_method_dat(system_root / "MINES" / combo["label"] / "processed" / "aggregate.dat", xs, aggregate)
        metrics = evaluate_combo(aggregate, xs, eval_xs, analytic_eval)
        metrics["label"] = combo["label"]
        mines_rankings.append(combo_row(metrics, combo, False))
    if mines_rankings:
        mines_rankings.sort(key=selection_sort_key)
        mines_selection = mines_rankings[0]
        for row in mines_rankings:
            row["selected"] = "true" if row["label"] == mines_selection["label"] else "false"
        write_ranking_csv(system_root / "MINES" / "rankings" / "all.csv", mines_rankings)
        selected_mines = read_method_dat(system_root / "MINES" / mines_selection["label"] / "processed" / "aggregate.dat")
        write_method_dat(selected_root / "mines.dat", xs, selected_mines)
    else:
        mines_selection = {"label": "missing", "status": "missing"}
        write_method_dat(selected_root / "mines.dat", xs, [])

    mtd_rankings: list[dict] = []
    mtd_selection: dict | None = None
    for combo in ctx["mtd_screen"]["combos"]:
        reps = load_seed_replicates(
            system_root / "MTD" / combo["label"] / "processed" / f"seed_{seed}.dat"
            for seed in ctx["seeds"]
        )
        aggregate = aggregate_replicates(reps)
        write_method_dat(system_root / "MTD" / combo["label"] / "processed" / "aggregate.dat", xs, aggregate)
        metrics = evaluate_combo(aggregate, xs, eval_xs, analytic_eval)
        metrics["label"] = combo["label"]
        mtd_rankings.append(combo_row(metrics, combo, False))
    mtd_rankings.sort(key=selection_sort_key)
    mtd_selection = mtd_rankings[0]
    for row in mtd_rankings:
        row["selected"] = "true" if row["label"] == mtd_selection["label"] else "false"
    write_ranking_csv(system_root / "MTD" / "rankings" / "all.csv", mtd_rankings)
    selected_mtd = read_method_dat(system_root / "MTD" / mtd_selection["label"] / "processed" / "aggregate.dat")
    write_method_dat(selected_root / "mtd.dat", xs, selected_mtd)

    selection = {
        "US": us_selection,
        "AUS": aus_selection,
        "NES": nes_selection,
        "MINES": mines_selection,
        "MTD": mtd_selection,
    }
    write_summary(
        selected_root / "summary.dat",
        system_root,
        ctx,
        selection,
        us_rankings,
        aus_rankings,
        nes_rankings,
        mines_rankings,
        mtd_rankings,
    )
    write_json(selected_root / "selection.json", selection)

    selected_us_gif = system_root / "US" / us_selection["label"] / "gifs" / f"seed_{ctx['seeds'][0]}.gif"
    if selected_us_gif.exists():
        (gifs_root).mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_us_gif, gifs_root / "us.gif")
    selected_mtd_gif = system_root / "MTD" / mtd_selection["label"] / "gifs" / f"seed_{ctx['seeds'][0]}.gif"
    if selected_mtd_gif.exists():
        gifs_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_mtd_gif, gifs_root / "mtd.gif")
    nes_gif_src = system_root / "NES" / nes_selection["label"] / "gifs"
    if nes_gif_src.exists():
        nes_gif_dst = gifs_root / "nes"
        nes_gif_dst.mkdir(parents=True, exist_ok=True)
        for gif_path in sorted(nes_gif_src.glob("*.gif")):
            shutil.copy2(gif_path, nes_gif_dst / gif_path.name)


def main() -> None:
    args = parse_args()
    system_root = Path(args.system_root).resolve()
    ctx = normalize_context(load_json(system_root / "run_context.json"))

    if args.command == "process-us-seed":
        process_us_seed(system_root, ctx, args.combo_label, args.seed, args.make_gif)
        return
    if args.command == "process-aus-seed":
        process_aus_seed(system_root, ctx, args.combo_label, args.seed)
        return
    if args.command == "process-mtd-seed":
        process_mtd_seed(system_root, ctx, args.combo_label, args.seed, args.make_gif)
        return
    if args.command == "process-nes-seed-time":
        process_nes_seed_time(
            system_root,
            ctx,
            args.combo_label,
            args.seed,
            args.time_steps,
            args.make_gif,
            args.retain_reduced,
        )
        return
    if args.command == "process-mines-seed":
        process_mines_seed(system_root, ctx, args.combo_label, args.seed)
        return
    if args.command == "finalize":
        finalize(system_root, ctx)
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
