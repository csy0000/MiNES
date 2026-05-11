#!/usr/bin/env python3
"""Generate static comparison figures for one benchmark system root.

The notebook consumes these PNGs directly, so this script focuses on stable
summary views rather than exploratory plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = REPO_ROOT / ".matplotlib-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
# Point Matplotlib at a repo-local cache so headless plotting does not depend
# on user-global state.
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("US", "AUS", "NES", "MINES", "MTD")
FILE_MAP = {
    "US": "us.dat",
    "AUS": "aus.dat",
    "NES": "nes.dat",
    "MINES": "mines.dat",
    "MTD": "mtd.dat",
}
METHOD_COLORS = {
    "US": "#0b6e4f",
    "AUS": "#5b8c5a",
    "NES": "#c84c09",
    "MINES": "#7a5195",
    "MTD": "#2f4858",
}
DEFAULT_SYSTEM_SLUG = "DoubleWell__k0_1p0__x0_m10p0__k1_1p0__x1_10p0__E1_10p0__kT_1p0__dt_0p0005__gamma_1p0"
DEFAULT_SYSTEM_ROOT = REPO_ROOT / "data" / "1D" / DEFAULT_SYSTEM_SLUG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system-root",
        default=os.environ.get("BENCHMARK_SYSTEM_ROOT", str(DEFAULT_SYSTEM_ROOT)),
        help="Benchmark system root under data/1D/<system_slug>.",
    )
    parser.add_argument(
        "--tag",
        default="doublewell_benchmark",
        help="Filename prefix for generated figures.",
    )
    return parser.parse_args()


def load_grid_table(path: Path) -> dict[str, np.ndarray]:
    # The reducer writes one flat CSV per method. Rebuild the regular
    # time-by-position arrays here for plotting.
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"{path} is empty")
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)
    times = np.unique(data["time"])
    xs = np.unique(data["x"])
    n_time = len(times)
    n_x = len(xs)
    order = np.lexsort((data["x"], data["time"]))
    ordered = data[order]
    f_est = ordered["F_est"].reshape(n_time, n_x)
    var_est = ordered["var_est"].reshape(n_time, n_x)
    return {
        "times": times,
        "x": xs,
        "F_est": f_est,
        "var_est": var_est,
        "sigma": np.sqrt(np.clip(var_est, 0.0, None)),
    }


def load_context(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_ranking_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = dict(row)
        for key in (
            "final_time",
            "central_coverage_fraction",
            "final_rmse",
            "final_mean_var",
            "time_avg_rmse",
            "k",
            "dx",
            "k_pull",
            "biasfactor",
        ):
            if key in item and item[key] != "":
                item[key] = float(item[key])
        parsed.append(item)
    return parsed


def analytic_doublewell_profile(xs: np.ndarray, ctx: dict) -> np.ndarray:
    pot = ctx["potential"]
    kT = float(ctx["thermal_kT"])
    beta = 1.0 / kT
    u0 = pot["k0"] * (xs - pot["x0"]) ** 2
    u1 = pot["k1"] * (xs - pot["x1"]) ** 2
    log_t0 = -beta * u0
    log_t1 = -beta * u1 - pot["E1"]
    log_max = np.maximum(log_t0, log_t1)
    free_energy = -(log_max + np.log(np.exp(log_t0 - log_max) + np.exp(log_t1 - log_max))) / beta
    return free_energy - np.nanmin(free_energy)


def build_eval_grid(ctx: dict) -> np.ndarray:
    grid = ctx["rmse_eval_grid"]
    xmin = float(grid["xmin"])
    xmax = float(grid["xmax"])
    dx = float(grid["dx"])
    n = int(round((xmax - xmin) / dx))
    return np.asarray([xmin + dx * i for i in range(n + 1)], dtype=float)


def sample_profile_on_grid(source_x: np.ndarray, source_values: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    lookup = {
        round(float(x), 10): float(value)
        for x, value in zip(source_x, source_values)
        if np.isfinite(value)
    }
    sampled = np.full(len(target_x), np.nan, dtype=float)
    for idx, x in enumerate(target_x):
        value = lookup.get(round(float(x), 10))
        if value is not None and np.isfinite(value):
            sampled[idx] = value
    return sampled


def rmse_curve(grid: dict[str, np.ndarray], ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    eval_x = build_eval_grid(ctx)
    analytic_eval = analytic_doublewell_profile(eval_x, ctx)
    rmses = np.full(len(grid["times"]), np.nan, dtype=float)
    for idx in range(len(grid["times"])):
        # RMSE is only evaluated where a method actually produced a finite PMF
        # estimate on the common comparison grid.
        sampled = sample_profile_on_grid(grid["x"], grid["F_est"][idx], eval_x)
        finite = np.isfinite(sampled)
        if np.any(finite):
            rmses[idx] = float(np.sqrt(np.mean((sampled[finite] - analytic_eval[finite]) ** 2)))
    return grid["times"], rmses


def combo_grids(system_root: Path, method: str, ctx: dict) -> dict[str, dict[str, np.ndarray]]:
    if method == "US":
        combos = ctx["us_screen"]["combos"]
    elif method == "AUS":
        combos = ctx["aus_screen"]["combos"]
    elif method == "NES":
        combos = ctx["nes_screen"]["combos"]
    elif method == "MINES":
        combos = ctx["mines_screen"]["combos"]
    else:
        combos = ctx["mtd_screen"]["combos"]
    grids: dict[str, dict[str, np.ndarray]] = {}
    for combo in combos:
        # Missing or empty aggregate files are expected for methods that were
        # not run in a given system root.
        path = system_root / method / combo["label"] / "processed" / "aggregate.dat"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            grids[combo["label"]] = load_grid_table(path)
        except ValueError:
            continue
    return grids


def plot_method_rmse(
    method: str,
    combo_grids_map: dict[str, dict[str, np.ndarray]],
    selection: dict,
    ctx: dict,
    out_path: Path,
) -> None:
    if not combo_grids_map:
        return
    selected_label = selection[method]["label"]
    # Highlight the selected screening winner while still showing the rest of
    # the parameter sweep for context.
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    if method == "US":
        cmap = plt.cm.viridis
    elif method == "AUS":
        cmap = plt.cm.Greens
    elif method == "NES":
        cmap = plt.cm.plasma
    elif method == "MINES":
        cmap = plt.cm.magma
    else:
        cmap = plt.cm.cividis
    labels = sorted(combo_grids_map.keys())
    colors = cmap(np.linspace(0.15, 0.9, len(labels)))
    for color, label in zip(colors, labels):
        times, rmses = rmse_curve(combo_grids_map[label], ctx)
        linewidth = 2.8 if label == selected_label else 1.6
        alpha = 1.0 if label == selected_label else 0.55
        zorder = 3 if label == selected_label else 2
        ax.plot(times, rmses, color=color, lw=linewidth, alpha=alpha, label=label, zorder=zorder)
    ax.set_xscale("log")
    ax.set_xlabel("Total simulation steps")
    ax.set_ylabel("RMSE on x in [-10, 10], dx = 0.2")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title(f"{method} parameter screening")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_best_rmse(best_grids: dict[str, dict[str, np.ndarray]], ctx: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    for method in METHODS:
        if method not in best_grids:
            continue
        times, rmses = rmse_curve(best_grids[method], ctx)
        ax.plot(times, rmses, color=METHOD_COLORS[method], lw=2.4, label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Total simulation steps")
    ax.set_ylabel("RMSE on x in [-10, 10], dx = 0.2")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Best parameter set from each method")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_best_final_profiles(best_grids: dict[str, dict[str, np.ndarray]], ctx: dict, out_path: Path) -> None:
    methods = [method for method in METHODS if method in best_grids]
    if not methods:
        return
    x_ref = best_grids[methods[0]]["x"]
    analytic = analytic_doublewell_profile(x_ref, ctx)
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    ax.plot(x_ref, analytic, color="#111111", linestyle="--", lw=2.2, label="Analytic")
    for method in methods:
        grid = best_grids[method]
        # Shift each final profile to its own minimum so the shape comparison is
        # not dominated by an arbitrary additive constant.
        y = grid["F_est"][-1] - np.nanmin(grid["F_est"][-1])
        sigma = grid["sigma"][-1]
        ax.plot(grid["x"], y, color=METHOD_COLORS[method], lw=2.0, label=method)
        ax.fill_between(grid["x"], y - sigma, y + sigma, color=METHOD_COLORS[method], alpha=0.18)
    ax.set_xlabel("x")
    ax.set_ylabel("Final PMF shifted to min = 0")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Best final PMFs vs analytic solution")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    system_root = Path(args.system_root).resolve()
    selected_root = system_root / "benchmark" / "selected"
    figures_root = system_root / "benchmark" / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    ctx = load_context(system_root / "run_context.json")
    selection = load_json(selected_root / "selection.json")

    method_combo_grids = {method: combo_grids(system_root, method, ctx) for method in METHODS}
    best_grids: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        selected_label = selection[method]["label"]
        grid = method_combo_grids[method].get(selected_label)
        if grid is not None:
            best_grids[method] = grid

    plot_method_rmse("US", method_combo_grids["US"], selection, ctx, figures_root / f"{args.tag}_us_rmse.png")
    plot_method_rmse("AUS", method_combo_grids["AUS"], selection, ctx, figures_root / f"{args.tag}_aus_rmse.png")
    plot_method_rmse("NES", method_combo_grids["NES"], selection, ctx, figures_root / f"{args.tag}_nes_rmse.png")
    plot_method_rmse("MINES", method_combo_grids["MINES"], selection, ctx, figures_root / f"{args.tag}_mines_rmse.png")
    plot_method_rmse("MTD", method_combo_grids["MTD"], selection, ctx, figures_root / f"{args.tag}_mtd_rmse.png")
    plot_best_rmse(best_grids, ctx, figures_root / f"{args.tag}_best_rmse.png")
    plot_best_final_profiles(best_grids, ctx, figures_root / f"{args.tag}_best_final_profiles.png")


if __name__ == "__main__":
    main()
