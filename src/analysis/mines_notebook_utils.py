"""Shared helpers for MiNES notebook builders and executed notebooks."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_bar(result):
    if isinstance(result, dict):
        return float(result["Delta_f"]), float(result.get("dDelta_f", float("nan")))
    if isinstance(result, tuple):
        return float(result[0]), float(result[1]) if len(result) > 1 else float("nan")
    return float(result), float("nan")


def read_path_rows(path):
    rows = pd.read_csv(path)
    return rows.rename(columns={"x0": "center_x"})[["lambda", "center_x", "k"]]


def load_trajectories(traj_paths):
    return [pd.read_csv(path) for path in traj_paths]


def neq_stage_cost_from_endpoints(forward_end_df, reverse_end_df):
    forward_steps = float(forward_end_df["final_step"].max()) + 1.0 if len(forward_end_df) else 0.0
    reverse_steps = float(reverse_end_df["final_step"].max()) + 1.0 if len(reverse_end_df) else 0.0
    return float(forward_steps + reverse_steps)


def rmse(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - ref[mask]) ** 2)))


def align_to_anchor(pmf, analytic, grid, anchor_x):
    pmf = np.asarray(pmf, dtype=float).copy()
    analytic = np.asarray(analytic, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 0:
        return pmf
    idx = int(np.argmin(np.abs(grid - float(anchor_x))))
    if not np.isfinite(pmf[idx]) or not np.isfinite(analytic[idx]):
        return pmf
    pmf[np.isfinite(pmf)] -= float(pmf[idx] - analytic[idx])
    return pmf


def align_to_value(pmf, grid, anchor_x, target_value):
    pmf = np.asarray(pmf, dtype=float).copy()
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 0:
        return pmf
    idx = int(np.argmin(np.abs(grid - float(anchor_x))))
    if not np.isfinite(pmf[idx]) or not np.isfinite(target_value):
        return pmf
    pmf[np.isfinite(pmf)] -= float(pmf[idx] - float(target_value))
    return pmf


def align_to_zero_at_anchor(pmf, grid, anchor_x):
    pmf = np.asarray(pmf, dtype=float).copy()
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 0:
        return pmf
    idx = int(np.argmin(np.abs(grid - float(anchor_x))))
    if not np.isfinite(pmf[idx]):
        return np.full(len(grid), np.nan, dtype=float)
    pmf[np.isfinite(pmf)] -= float(pmf[idx])
    return pmf


def align_to_reference_min_rmse(pmf, ref):
    pmf = np.asarray(pmf, dtype=float).copy()
    ref = np.asarray(ref, dtype=float)
    mask = np.isfinite(pmf) & np.isfinite(ref)
    if not np.any(mask):
        return pmf, float("nan")
    shift = float(np.nanmean(ref[mask] - pmf[mask]))
    pmf[np.isfinite(pmf)] += shift
    return pmf, shift


def value_at_x(arr, grid, x):
    arr = np.asarray(arr, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(grid - float(x))))
    value = float(arr[idx])
    return value if np.isfinite(value) else float("nan")


def masked_interval(arr, grid, start_x, end_x):
    arr = np.asarray(arr, dtype=float)
    grid = np.asarray(grid, dtype=float)
    lo = min(float(start_x), float(end_x)) - 1.0e-9
    hi = max(float(start_x), float(end_x)) + 1.0e-9
    out = np.full(len(arr), np.nan, dtype=float)
    mask = (grid >= lo) & (grid <= hi) & np.isfinite(arr)
    out[mask] = arr[mask]
    return out


def combine_contributions(contribs):
    if not contribs:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )
    stack = np.vstack([np.asarray(arr, dtype=float) for arr in contribs])
    counts = np.sum(np.isfinite(stack), axis=0).astype(float)
    mean = np.full(stack.shape[1], np.nan, dtype=float)
    var = np.full(stack.shape[1], np.nan, dtype=float)
    any_mask = counts > 0
    overlap_mask = counts > 1
    if np.any(any_mask):
        mean[any_mask] = np.nanmean(stack[:, any_mask], axis=0)
        var[any_mask] = 0.0
    if np.any(overlap_mask):
        var[overlap_mask] = np.nanvar(stack[:, overlap_mask], axis=0, ddof=0)
    return mean, var, counts


def build_edges_from_grid(grid):
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 0:
        return np.array([0.0, 1.0], dtype=float)
    if len(grid) == 1:
        half_width = 0.5
        return np.array([grid[0] - half_width, grid[0] + half_width], dtype=float)
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    left_edge = grid[0] - (midpoints[0] - grid[0])
    right_edge = grid[-1] + (grid[-1] - midpoints[-1])
    return np.concatenate([[left_edge], midpoints, [right_edge]])


def coverage_mask_from_samples(samples, grid):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    grid = np.asarray(grid, dtype=float)
    if samples.size == 0 or grid.size == 0:
        return np.zeros(len(grid), dtype=bool)
    edges = build_edges_from_grid(grid)
    counts, _ = np.histogram(samples, bins=edges)
    return counts > 0


def interval_coverage_mask(grid, start_x, end_x):
    grid = np.asarray(grid, dtype=float)
    lo = min(float(start_x), float(end_x)) - 1.0e-9
    hi = max(float(start_x), float(end_x)) + 1.0e-9
    return (grid >= lo) & (grid <= hi)


def mode_x_from_samples(samples, grid):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan")
    grid = np.asarray(grid, dtype=float)
    edges = build_edges_from_grid(grid)
    counts, _ = np.histogram(samples, bins=edges)
    if counts.size == 0 or int(np.max(counts)) <= 0:
        return float("nan")
    return float(grid[int(np.argmax(counts))])


def endpoint_extrema(samples):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def forward_work_values(df):
    if "appended_work" in df.columns:
        return df["appended_work"].to_numpy(dtype=float)
    if "final_work" in df.columns:
        return df["final_work"].to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)


def pair_dissipative_work(forward_work, reverse_work):
    forward_work = np.asarray(forward_work, dtype=float)
    reverse_work = np.asarray(reverse_work, dtype=float)
    if not np.any(np.isfinite(forward_work)) or not np.any(np.isfinite(reverse_work)):
        return float("nan")
    return float(0.5 * (np.nanmean(forward_work) + np.nanmean(reverse_work)))


def background_potential_1d(grid, run_context):
    grid = np.asarray(grid, dtype=float)
    potential = run_context["potential"]
    kT = float(run_context["thermal_kT"])
    u0 = float(potential["k0"]) * (grid - float(potential["x0"])) ** 2
    u1 = float(potential["k1"]) * (grid - float(potential["x1"])) ** 2
    log_t0 = -u0 / kT
    log_t1 = -u1 / kT - float(potential["E1"])
    log_max = np.maximum(log_t0, log_t1)
    log_sum = log_max + np.log(np.exp(log_t0 - log_max) + np.exp(log_t1 - log_max))
    background = -kT * log_sum
    background -= np.nanmin(background)
    return background


def window_sort_key(name):
    name = str(name)
    prefix = name[0] if name else ""
    if prefix == "M":
        match = re.match(r"^M(\d+)", name)
        if match:
            return (prefix, int(match.group(1)), name)
    try:
        suffix = int(name[1:])
    except ValueError:
        suffix = 999
    return (prefix, suffix, name)


def rescue_window_name(value):
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value or "")


def grid_index(x, grid):
    grid = np.asarray(grid, dtype=float)
    if grid.size == 0:
        return None
    if grid.size == 1:
        return 0 if abs(float(x) - float(grid[0])) <= 0.5 else None
    dx = abs(float(grid[1] - grid[0]))
    idx = int(np.rint((float(x) - float(grid[0])) / dx))
    if idx < 0 or idx >= len(grid):
        return None
    if abs(float(grid[idx]) - float(x)) > 0.5 * dx + 1.0e-9:
        return None
    return idx


def shift_finite_to_zero(values):
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if np.any(finite):
        values[finite] -= float(np.nanmin(values[finite]))
    return values


def hs_reconstruct_oneway_arrays(x, work, centers, k_values, grid, kT=1.0):
    x = np.asarray(x, dtype=float)
    work = np.asarray(work, dtype=float)
    centers = np.asarray(centers, dtype=float)
    k_values = np.asarray(k_values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if x.ndim != 2 or work.ndim != 2 or x.shape[0] == 0 or work.shape[0] == 0:
        return np.full(len(grid), np.nan, dtype=float)
    n_time = min(x.shape[1], work.shape[1], len(centers), len(k_values))
    if n_time <= 0 or len(grid) == 0:
        return np.full(len(grid), np.nan, dtype=float)
    x = x[:, :n_time]
    work = work[:, :n_time]
    centers = centers[:n_time]
    k_values = k_values[:n_time]
    beta = 1.0 / float(kT)
    dx = abs(float(grid[1] - grid[0])) if len(grid) > 1 else 1.0
    n_traj = x.shape[0]
    x_left = float(grid[0])
    weights = np.exp(-beta * work)
    sum_w = np.sum(weights, axis=0)
    idx = np.rint((x - x_left) / dx).astype(int)
    valid = (idx >= 0) & (idx < len(grid))
    if len(grid):
        clipped = np.clip(idx, 0, len(grid) - 1)
        valid &= np.abs(grid[clipped] - x) <= 0.5 * dx + 1.0e-9
    log_numerator_terms = np.full((n_time, len(grid)), -np.inf, dtype=float)
    log_n_traj = np.log(float(n_traj))
    for time_idx in range(n_time):
        if sum_w[time_idx] <= 0.0:
            continue
        valid_t = valid[:, time_idx]
        if not np.any(valid_t):
            continue
        hist_sum = np.bincount(
            idx[valid_t, time_idx],
            weights=weights[valid_t, time_idx],
            minlength=len(grid),
        ).astype(float)
        positive = hist_sum > 0.0
        if np.any(positive):
            log_numerator_terms[time_idx, positive] = (
                np.log(hist_sum[positive]) - log_n_traj - np.log(float(sum_w[time_idx]))
            )
    log_denominator_terms = np.full((n_time, len(grid)), -np.inf, dtype=float)
    valid_time = sum_w > 0.0
    if np.any(valid_time):
        grid_row = grid[None, :]
        log_denominator_terms[valid_time] = (
            -np.log(sum_w[valid_time])[:, None]
            - beta
            * 0.5
            * k_values[valid_time, None]
            * (grid_row - centers[valid_time, None])
            * (grid_row - centers[valid_time, None])
        )
    with np.errstate(invalid="ignore"):
        log_density = np.logaddexp.reduce(log_numerator_terms, axis=0) - np.logaddexp.reduce(
            log_denominator_terms, axis=0
        )
    valid_density = np.isfinite(log_density)
    if np.any(valid_density):
        finite_logs = log_density[valid_density]
        max_log = float(np.max(finite_logs))
        log_norm = max_log + np.log(np.sum(np.exp(finite_logs - max_log))) + np.log(dx)
        log_density[valid_density] -= log_norm
    pmf = np.full(len(grid), np.nan, dtype=float)
    pmf[valid_density] = -float(kT) * log_density[valid_density]
    return shift_finite_to_zero(pmf)


def bootstrap_oneway_hs_stack(
    x,
    work,
    centers,
    k_values,
    grid,
    anchor_x,
    n_boot,
    rng_seed,
):
    x = np.asarray(x, dtype=float)
    work = np.asarray(work, dtype=float)
    if x.ndim != 2 or work.ndim != 2 or x.shape[0] == 0 or work.shape[0] == 0:
        return np.empty((0, len(grid)), dtype=float)
    rng = np.random.default_rng(int(rng_seed))
    stack = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, x.shape[0], size=x.shape[0])
        pmf_boot = hs_reconstruct_oneway_arrays(
            x[idx],
            work[idx],
            centers,
            k_values,
            grid,
            kT=1.0,
        )
        pmf_boot = align_to_zero_at_anchor(pmf_boot, grid, anchor_x)
        if np.any(np.isfinite(pmf_boot)):
            stack.append(pmf_boot)
    if not stack:
        return np.empty((0, len(grid)), dtype=float)
    return np.vstack(stack)


def align_pmf_to_analytic_anchor(pmf, analytic, grid, anchor_x):
    aligned = np.asarray(pmf, dtype=float).copy()
    analytic = np.asarray(analytic, dtype=float)
    idx = grid_index(anchor_x, grid)
    if idx is None:
        return aligned
    if not np.isfinite(aligned[idx]) or not np.isfinite(analytic[idx]):
        return aligned
    aligned[np.isfinite(aligned)] -= float(aligned[idx] - analytic[idx])
    return aligned


def bootstrap_oneway_stack(x, work, centers, k_values, grid, analytic, anchor_x, n_boot, seed):
    rng = np.random.default_rng(seed)
    stack = []
    if x.ndim != 2 or work.ndim != 2 or x.shape[0] == 0 or work.shape[0] == 0:
        return np.empty((0, len(grid)), dtype=float)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, x.shape[0], size=x.shape[0])
        pmf_boot = hs_reconstruct_oneway_arrays(
            x[idx],
            work[idx],
            centers,
            k_values,
            grid,
            kT=1.0,
        )
        pmf_boot = align_pmf_to_analytic_anchor(pmf_boot, analytic, grid, anchor_x)
        if np.any(np.isfinite(pmf_boot)):
            stack.append(pmf_boot)
    if not stack:
        return np.empty((0, len(grid)), dtype=float)
    return np.vstack(stack)


def stack_summary(stack, grid):
    if stack.size == 0:
        nan = np.full(len(grid), np.nan, dtype=float)
        return nan.copy(), nan.copy(), nan.copy()
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        q05 = np.nanpercentile(stack, 5.0, axis=0)
        q95 = np.nanpercentile(stack, 95.0, axis=0)
    return mean, q05, q95


def _window_name_to_mathtext(name):
    name = str(name or "")
    rescue_match = re.fullmatch(r"([A-Za-z]+)(\d+)\^\[([^\]]+)\]_\{([^}]+)\}", name)
    if rescue_match:
        prefix, idx, upper, lower = rescue_match.groups()
        return (
            rf"{prefix}_{{{idx}}}"
            rf"^{{[{_window_name_to_mathtext(upper)}]_{{{_window_name_to_mathtext(lower)}}}}}"
        )
    simple_match = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if simple_match:
        prefix, idx = simple_match.groups()
        return rf"{prefix}_{{{idx}}}"
    return name.replace("_", r"\_")


def window_name_math(name):
    return f"${_window_name_to_mathtext(name)}$"


def pair_name_math(source_name, target_name, relation="leftrightarrow"):
    relation_map = {
        "leftrightarrow": r"\leftrightarrow",
        "rightarrow": r"\rightarrow",
        "leftarrow": r"\leftarrow",
    }
    relation_text = relation_map.get(str(relation), str(relation))
    return f"${_window_name_to_mathtext(source_name)}\\ {relation_text}\\ {_window_name_to_mathtext(target_name)}$"


def markdown_escape(text):
    text = str(text)
    return text.replace("|", r"\|").replace("\n", "<br>")


def format_markdown_value(value, digits=3):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(scalar):
        return ""
    if abs(scalar - round(scalar)) < 1.0e-9:
        return str(int(round(scalar)))
    return f"{scalar:.{digits}f}"


def markdown_table(headers, rows):
    escaped_headers = [markdown_escape(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)
