#!/usr/bin/env python3
"""Write per-method metadata files under a benchmark system root.

The shell runners store the full system context once in `run_context.json`.
This helper fans that out into method-local context files so downstream
analysis and notebook code can discover layouts without duplicating logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system-root",
        required=True,
        help="System root under data/1D/<system_slug>.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def base_context(ctx: dict) -> dict:
    # Keep the method-local files self-contained for browsing inside `data/`
    # without forcing every consumer to reopen the system root JSON.
    return {
        "system_name": ctx["system_name"],
        "system_slug": ctx["system_slug"],
        "potential_name": ctx["potential_name"],
        "one_dimension": ctx["one_dimension"],
        "basins": ctx["basins"],
        "potential": ctx["potential"],
        "thermal_kT": ctx["thermal_kT"],
        "dt": ctx["dt"],
        "gamma": ctx["gamma"],
        "time_grid": ctx["time_grid"],
    }


def write_us_contexts(system_root: Path, ctx: dict) -> None:
    us_root = system_root / "US"
    us_screen = ctx["us_screen"]
    method_context = {
        **base_context(ctx),
        "method": "US",
        "method_label": "Fixed-window umbrella sampling",
        "directional_scheme": "none",
        "method_root": "US",
        "files": {
            "method_context": "US/method_context.json",
            "combo_context": "US/<combo_label>/combo_context.json",
            "seed_processed": "US/<combo_label>/processed/seed_<seed>.dat",
            "aggregate_processed": "US/<combo_label>/processed/aggregate.dat",
            "reduced_seed_trajectory": "US/<combo_label>/reduced/seed_<seed>.csv",
            "gif": "US/<combo_label>/gifs/seed_101.gif",
            "ranking_csv": "US/rankings/all.csv",
        },
        "screen": us_screen,
    }
    write_json(us_root / "method_context.json", method_context)

    fixed = us_screen["fixed"]
    for combo in us_screen["combos"]:
        combo_context = {
            **base_context(ctx),
            "method": "US",
            "method_label": "Fixed-window umbrella sampling",
            "combo_label": combo["label"],
            "combo_root": f"US/{combo['label']}",
            "parameters": {
                "k": combo["k"],
                "dx": combo["dx"],
                "n_windows": combo["n_windows"],
                "steps_per_window": combo["steps_per_window"],
                "remainder_windows": combo["remainder_windows"],
                "total_steps": combo["total_steps"],
                "sample_stride_steps": combo["sample_stride_steps"],
                "output_samples": combo["output_samples"],
            },
            "fixed": fixed,
            "files": {
                "seed_processed": "processed/seed_<seed>.dat",
                "aggregate_processed": "processed/aggregate.dat",
                "reduced_seed_trajectory": "reduced/seed_<seed>.csv",
                "gif": "gifs/seed_101.gif",
            },
        }
        write_json(us_root / combo["label"] / "combo_context.json", combo_context)


def write_aus_contexts(system_root: Path, ctx: dict) -> None:
    aus_root = system_root / "AUS"
    aus_screen = ctx["aus_screen"]
    method_context = {
        **base_context(ctx),
        "method": "AUS",
        "method_label": "Adaptive umbrella sampling",
        "directional_scheme": "paired left/right adaptive growth from both basins",
        "method_root": "AUS",
        "files": {
            "method_context": "AUS/method_context.json",
            "combo_context": "AUS/<combo_label>/combo_context.json",
            "seed_processed": "AUS/<combo_label>/processed/seed_<seed>.dat",
            "aggregate_processed": "AUS/<combo_label>/processed/aggregate.dat",
            "reduced_seed_trajectory": "AUS/<combo_label>/reduced/seed_<seed>.csv",
            "ranking_csv": "AUS/rankings/all.csv",
        },
        "screen": aus_screen,
    }
    write_json(aus_root / "method_context.json", method_context)

    fixed = aus_screen["fixed"]
    for combo in aus_screen["combos"]:
        combo_context = {
            **base_context(ctx),
            "method": "AUS",
            "method_label": "Adaptive umbrella sampling",
            "combo_label": combo["label"],
            "combo_root": f"AUS/{combo['label']}",
            "parameters": {
                "q_next": combo["q_next"],
                "alpha": combo["alpha"],
                "fit_method": combo.get("fit_method", ""),
                "k_min": combo["k_min"],
                "k_max": combo["k_max"],
            },
            "fixed": fixed,
            "files": {
                "seed_processed": "processed/seed_<seed>.dat",
                "aggregate_processed": "processed/aggregate.dat",
                "reduced_seed_trajectory": "reduced/seed_<seed>.csv",
            },
        }
        write_json(aus_root / combo["label"] / "combo_context.json", combo_context)


def write_nes_contexts(system_root: Path, ctx: dict) -> None:
    nes_root = system_root / "NES"
    nes_screen = ctx["nes_screen"]
    method_context = {
        **base_context(ctx),
        "method": "NES",
        "method_label": "Bidirectional nonequilibrium switching",
        "directional_scheme": "bidirectional only",
        "method_root": "NES",
        "files": {
            "method_context": "NES/method_context.json",
            "combo_context": "NES/<combo_label>/combo_context.json",
            "seed_processed": "NES/<combo_label>/processed/seed_<seed>.dat",
            "aggregate_processed": "NES/<combo_label>/processed/aggregate.dat",
            "reduced_seed_trajectory": "NES/<combo_label>/reduced/seed_<seed>.csv",
            "gif": "NES/<combo_label>/gifs/T_<steps>.gif",
            "ranking_csv": "NES/rankings/all.csv",
        },
        "screen": nes_screen,
    }
    write_json(nes_root / "method_context.json", method_context)

    fixed = nes_screen["fixed"]
    for combo in nes_screen["combos"]:
        combo_context = {
            **base_context(ctx),
            "method": "NES",
            "method_label": "Bidirectional nonequilibrium switching",
            "combo_label": combo["label"],
            "combo_root": f"NES/{combo['label']}",
            "parameters": {
                "k": combo["k"],
                "k_midscale": combo["k_midscale"],
            },
            "fixed": fixed,
            "files": {
                "seed_processed": "processed/seed_<seed>.dat",
                "aggregate_processed": "processed/aggregate.dat",
                "reduced_seed_trajectory": "reduced/seed_<seed>.csv",
                "gif": "gifs/T_<steps>.gif",
            },
        }
        write_json(nes_root / combo["label"] / "combo_context.json", combo_context)


def write_mines_contexts(system_root: Path, ctx: dict) -> None:
    mines_root = system_root / "MINES"
    mines_screen = ctx["mines_screen"]
    method_context = {
        **base_context(ctx),
        "method": "MINES",
        "method_label": "Milestone-based nonequilibrium switching",
        "directional_scheme": "bidirectional adaptive milestone growth from endpoints",
        "method_root": "MINES",
        "files": {
            "method_context": "MINES/method_context.json",
            "combo_context": "MINES/<combo_label>/combo_context.json",
            "seed_processed": "MINES/<combo_label>/processed/seed_<seed>.dat",
            "aggregate_processed": "MINES/<combo_label>/processed/aggregate.dat",
            "reduced_seed_trajectory": "MINES/<combo_label>/reduced/seed_<seed>.json",
            "ranking_csv": "MINES/rankings/all.csv",
        },
        "screen": mines_screen,
    }
    write_json(mines_root / "method_context.json", method_context)

    fixed = mines_screen["fixed"]
    for combo in mines_screen["combos"]:
        combo_context = {
            **base_context(ctx),
            "method": "MINES",
            "method_label": "Milestone-based nonequilibrium switching",
            "combo_label": combo["label"],
            "combo_root": f"MINES/{combo['label']}",
            "parameters": {
                "k_pull": combo["k_pull"],
            },
            "fixed": fixed,
            "files": {
                "seed_processed": "processed/seed_<seed>.dat",
                "aggregate_processed": "processed/aggregate.dat",
                "reduced_seed_trajectory": "reduced/seed_<seed>.json",
            },
        }
        write_json(mines_root / combo["label"] / "combo_context.json", combo_context)


def normalize_context(ctx: dict) -> dict:
    # Preserve compatibility with older single-combo MTD contexts by promoting
    # them into the newer screened layout on the fly.
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
    return ctx


def write_mtd_context(system_root: Path, ctx: dict) -> None:
    mtd_root = system_root / "MTD"
    mtd_screen = ctx["mtd_screen"]
    method_context = {
        **base_context(ctx),
        "method": "MTD",
        "method_label": "Two-walker well-tempered metadynamics",
        "directional_scheme": "two walkers from both basins",
        "method_root": "MTD",
        "screen": mtd_screen,
        "files": {
            "method_context": "MTD/method_context.json",
            "combo_context": "MTD/<combo_label>/combo_context.json",
            "seed_processed": "MTD/<combo_label>/processed/seed_<seed>.dat",
            "aggregate_processed": "MTD/<combo_label>/processed/aggregate.dat",
            "reduced_seed_trajectory": "MTD/<combo_label>/reduced/seed_<seed>.csv",
            "gif": "MTD/<combo_label>/gifs/seed_101.gif",
        },
    }
    write_json(mtd_root / "method_context.json", method_context)

    fixed = mtd_screen["fixed"]
    for combo in mtd_screen["combos"]:
        combo_context = {
            **base_context(ctx),
            "method": "MTD",
            "method_label": "Two-walker well-tempered metadynamics",
            "combo_label": combo["label"],
            "combo_root": f"MTD/{combo['label']}",
            "parameters": {
                "biasfactor": combo["biasfactor"],
            },
            "fixed": fixed,
            "files": {
                "seed_processed": "processed/seed_<seed>.dat",
                "aggregate_processed": "processed/aggregate.dat",
                "reduced_seed_trajectory": "reduced/seed_<seed>.csv",
                "gif": "gifs/seed_101.gif",
            },
        }
        write_json(mtd_root / combo["label"] / "combo_context.json", combo_context)


def main() -> None:
    args = parse_args()
    system_root = Path(args.system_root).resolve()
    ctx = normalize_context(load_json(system_root / "run_context.json"))
    # Rebuild every method context together so the filesystem contract stays in
    # sync after any runner-side context change.
    write_us_contexts(system_root, ctx)
    write_aus_contexts(system_root, ctx)
    write_nes_contexts(system_root, ctx)
    write_mines_contexts(system_root, ctx)
    write_mtd_context(system_root, ctx)


if __name__ == "__main__":
    main()
