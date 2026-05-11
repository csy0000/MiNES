"""Shared current-protocol PMF analysis helpers for MiNES notebooks."""

from __future__ import annotations

import numpy as np
from pymbar import MBAR
from scipy.optimize import minimize

from bidirectional_mts_pmf import align_pmf_to_endpoint_average_zero
from mines_notebook_utils import (
    align_to_anchor,
    align_to_reference_min_rmse,
    align_to_value,
    align_to_zero_at_anchor,
    background_potential_1d,
    build_edges_from_grid,
    combine_contributions,
    coverage_mask_from_samples,
    interval_coverage_mask,
    masked_interval,
    rmse,
    value_at_x,
)


def reduced_doublewell_potential_np(values, run_context):
    values = np.asarray(values, dtype=float)
    potential = run_context["potential"]
    beta = 1.0 / float(run_context["thermal_kT"])
    u0 = float(potential["k0"]) * (values - float(potential["x0"])) ** 2
    u1 = float(potential["k1"]) * (values - float(potential["x1"])) ** 2
    log_t0 = -beta * u0
    log_t1 = -beta * u1 - float(potential["E1"])
    log_max = np.maximum(log_t0, log_t1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return -(log_max + np.log(np.exp(log_t0 - log_max) + np.exp(log_t1 - log_max)))


def direct_eq_mbar_pmf(window_rows, grid, run_context):
    nonempty = []
    for row in window_rows:
        samples = np.asarray(row["tail_x"], dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size <= 0:
            continue
        nonempty.append((samples, float(row["x_m"]), float(row["k_m"])))
    if not nonempty:
        return (
            np.full(len(grid), np.nan, dtype=float),
            np.zeros(len(grid), dtype=float),
            np.zeros(len(grid), dtype=float),
        )

    sample_arrays = [item[0] for item in nonempty]
    centers = np.asarray([item[1] for item in nonempty], dtype=float)
    ks = np.asarray([item[2] for item in nonempty], dtype=float)
    counts = np.asarray([arr.size for arr in sample_arrays], dtype=int)
    samples = np.concatenate(sample_arrays)

    beta = 1.0 / float(run_context["thermal_kT"])
    reduced_unbiased = reduced_doublewell_potential_np(samples, run_context)
    reduced_biased = (
        reduced_unbiased[None, :]
        + 0.5 * beta * ks[:, None] * (samples[None, :] - centers[:, None]) ** 2
    )
    u_kn = np.vstack([reduced_biased, reduced_unbiased[None, :]])
    N_k = np.concatenate([counts, np.asarray([0], dtype=int)])
    mbar = MBAR(
        u_kn,
        N_k,
        maximum_iterations=200,
        relative_tolerance=1.0e-6,
        verbose=False,
        initialize="zeros",
    )
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
    ess[positive_weight_sq] = (
        probability[positive_weight_sq] * probability[positive_weight_sq]
    ) / weight_squares[positive_weight_sq]

    free_energy = np.full(len(grid), np.nan, dtype=float)
    positive = probability > 0.0
    free_energy[positive] = -float(run_context["thermal_kT"]) * np.log(probability[positive])
    finite = np.isfinite(free_energy)
    if np.any(finite):
        free_energy[finite] -= np.min(free_energy[finite])
    return free_energy, ess, probability


def bootstrap_direct_eq_mbar(window_rows, grid, reference_x, rng_seed, run_context, n_boot):
    analytic_ref = background_potential_1d(grid, run_context)
    pmf, ess, probability = direct_eq_mbar_pmf(window_rows, grid, run_context)
    pmf = align_to_anchor(pmf, analytic_ref, grid, reference_x)

    rng = np.random.default_rng(int(rng_seed))
    boot_curves = []
    for _ in range(int(n_boot)):
        boot_rows = []
        for row in window_rows:
            samples = np.asarray(row["tail_x"], dtype=float)
            samples = samples[np.isfinite(samples)]
            if samples.size <= 0:
                continue
            boot_rows.append(
                {
                    "tail_x": rng.choice(samples, size=samples.size, replace=True),
                    "x_m": float(row["x_m"]),
                    "k_m": float(row["k_m"]),
                }
            )
        boot_pmf, _, _ = direct_eq_mbar_pmf(boot_rows, grid, run_context)
        boot_curves.append(align_to_anchor(boot_pmf, analytic_ref, grid, reference_x))

    boot_stack = (
        np.vstack(boot_curves)
        if boot_curves
        else np.full((0, len(grid)), np.nan, dtype=float)
    )
    if boot_stack.shape[0] > 0:
        with np.errstate(invalid="ignore"):
            boot_var = np.nanvar(
                boot_stack,
                axis=0,
                ddof=1 if boot_stack.shape[0] > 1 else 0,
            )
    else:
        boot_var = np.full(len(grid), np.nan, dtype=float)
    return {
        "pmf": pmf,
        "analytic": analytic_ref,
        "ess": ess,
        "probability": probability,
        "boot_stack": boot_stack,
        "boot_var": boot_var,
        "n_boot_used": int(boot_stack.shape[0]),
    }


def logistic_sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def estimate_multistate_nes_free_energies(edge_records, reference_state=None, initial_state_values=None):
    normalized_edges = []
    state_names = []
    for record in edge_records:
        source_name = str(record["source_name"])
        target_name = str(record["target_name"])
        forward_work = np.asarray(record["forward_work"], dtype=float)
        reverse_work = np.asarray(record["reverse_work"], dtype=float)
        forward_work = forward_work[np.isfinite(forward_work)]
        reverse_work = reverse_work[np.isfinite(reverse_work)]
        if forward_work.size == 0 or reverse_work.size == 0:
            continue
        normalized_edges.append(
            {
                "source_name": source_name,
                "target_name": target_name,
                "forward_work": forward_work,
                "reverse_work": reverse_work,
            }
        )
        if source_name not in state_names:
            state_names.append(source_name)
        if target_name not in state_names:
            state_names.append(target_name)
    if not normalized_edges or not state_names:
        raise RuntimeError("No finite bidirectional work data available for multistate MLE.")

    reference_state = str(reference_state or state_names[0])
    if reference_state not in state_names:
        state_names.insert(0, reference_state)
    free_states = [state for state in state_names if state != reference_state]
    state_index = {state: idx for idx, state in enumerate(free_states)}
    initial_state_values = initial_state_values or {}

    def unpack_state_values(x):
        state_values = {reference_state: 0.0}
        for state, value in zip(free_states, np.asarray(x, dtype=float)):
            state_values[state] = float(value)
        return state_values

    def objective_and_gradient(x):
        state_values = unpack_state_values(x)
        objective = 0.0
        gradient = np.zeros(len(free_states), dtype=float)
        for edge in normalized_edges:
            delta_f = float(state_values[edge["target_name"]] - state_values[edge["source_name"]])
            log_count_ratio = float(np.log(edge["forward_work"].size / edge["reverse_work"].size))
            z_forward = delta_f - log_count_ratio - edge["forward_work"]
            z_reverse = log_count_ratio - delta_f - edge["reverse_work"]
            objective += float(
                np.sum(np.logaddexp(0.0, z_forward))
                + np.sum(np.logaddexp(0.0, z_reverse))
            )
            d_delta = float(
                np.sum(logistic_sigmoid(z_forward)) - np.sum(logistic_sigmoid(z_reverse))
            )
            if edge["source_name"] != reference_state:
                gradient[state_index[edge["source_name"]]] -= d_delta
            if edge["target_name"] != reference_state:
                gradient[state_index[edge["target_name"]]] += d_delta
        return objective, gradient

    def hessian_from_state_values(state_values):
        hessian = np.zeros((len(free_states), len(free_states)), dtype=float)
        for edge in normalized_edges:
            delta_f = float(state_values[edge["target_name"]] - state_values[edge["source_name"]])
            log_count_ratio = float(np.log(edge["forward_work"].size / edge["reverse_work"].size))
            z_forward = delta_f - log_count_ratio - edge["forward_work"]
            z_reverse = log_count_ratio - delta_f - edge["reverse_work"]
            curvature = float(
                np.sum(logistic_sigmoid(z_forward) * (1.0 - logistic_sigmoid(z_forward)))
                + np.sum(logistic_sigmoid(z_reverse) * (1.0 - logistic_sigmoid(z_reverse)))
            )
            direction = np.zeros(len(free_states), dtype=float)
            if edge["source_name"] != reference_state:
                direction[state_index[edge["source_name"]]] -= 1.0
            if edge["target_name"] != reference_state:
                direction[state_index[edge["target_name"]]] += 1.0
            hessian += curvature * np.outer(direction, direction)
        return hessian

    if free_states:
        x0 = np.asarray(
            [float(initial_state_values.get(state, 0.0)) for state in free_states],
            dtype=float,
        )
        result = minimize(
            lambda x: objective_and_gradient(x)[0],
            x0,
            jac=lambda x: objective_and_gradient(x)[1],
            method="BFGS",
        )
        if not bool(result.success):
            result = minimize(
                lambda x: objective_and_gradient(x)[0],
                x0,
                method="Powell",
            )
        state_values = unpack_state_values(result.x)
        objective = float(result.fun)
    else:
        result = None
        state_values = {reference_state: 0.0}
        objective = 0.0

    hessian = hessian_from_state_values(state_values)
    if hessian.size:
        covariance = np.linalg.pinv(hessian)
    else:
        covariance = np.zeros((0, 0), dtype=float)
    state_std_hessian = {reference_state: 0.0}
    for state in free_states:
        variance = (
            float(covariance[state_index[state], state_index[state]])
            if covariance.size
            else float("nan")
        )
        state_std_hessian[state] = (
            float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else float("nan")
        )
    edge_delta_f = {
        f"{edge['source_name']} -> {edge['target_name']}": float(
            state_values[edge["target_name"]] - state_values[edge["source_name"]]
        )
        for edge in normalized_edges
    }

    return {
        "reference_state": reference_state,
        "states": list(state_names),
        "state_values": {state: float(state_values[state]) for state in state_names},
        "state_std_hessian": state_std_hessian,
        "covariance": np.asarray(covariance, dtype=float),
        "edge_delta_f": edge_delta_f,
        "negative_log_likelihood": float(objective),
        "success": True if result is None else bool(result.success),
        "message": "" if result is None else str(result.message),
    }


def bootstrap_multistate_nes_free_energies(
    edge_records,
    reference_state=None,
    n_boot=0,
    rng_seed=0,
    initial_state_values=None,
):
    base = estimate_multistate_nes_free_energies(
        edge_records,
        reference_state=reference_state,
        initial_state_values=initial_state_values,
    )
    state_names = list(base["states"])
    rng = np.random.default_rng(int(rng_seed))
    boot_rows = []
    for _ in range(int(n_boot)):
        boot_edges = []
        for record in edge_records:
            forward_work = np.asarray(record["forward_work"], dtype=float)
            reverse_work = np.asarray(record["reverse_work"], dtype=float)
            forward_work = forward_work[np.isfinite(forward_work)]
            reverse_work = reverse_work[np.isfinite(reverse_work)]
            if forward_work.size == 0 or reverse_work.size == 0:
                continue
            boot_record = dict(record)
            boot_record["forward_work"] = rng.choice(
                forward_work,
                size=forward_work.size,
                replace=True,
            )
            boot_record["reverse_work"] = rng.choice(
                reverse_work,
                size=reverse_work.size,
                replace=True,
            )
            boot_edges.append(boot_record)
        if not boot_edges:
            continue
        try:
            estimate = estimate_multistate_nes_free_energies(
                boot_edges,
                reference_state=base["reference_state"],
                initial_state_values=base["state_values"],
            )
        except Exception:
            continue
        boot_rows.append([float(estimate["state_values"][state]) for state in state_names])

    boot_stack = (
        np.asarray(boot_rows, dtype=float)
        if boot_rows
        else np.empty((0, len(state_names)), dtype=float)
    )
    boot_state_values = {
        state: boot_stack[:, idx].copy() if boot_stack.size else np.empty((0,), dtype=float)
        for idx, state in enumerate(state_names)
    }
    boot_state_std = {}
    for state, values in boot_state_values.items():
        if values.size > 0:
            ddof = 1 if values.size > 1 else 0
            with np.errstate(invalid="ignore"):
                variance = float(np.nanvar(values, ddof=ddof))
            boot_state_std[state] = (
                float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else float("nan")
            )
        else:
            boot_state_std[state] = float("nan")

    base["boot_state_values"] = boot_state_values
    base["boot_state_std"] = boot_state_std
    base["n_boot_used"] = int(boot_stack.shape[0])
    return base


def build_stitched_bundle(
    stitch_sequence,
    anchor_name,
    anchor_x,
    state_relative_values=None,
    state_boot_relative_values=None,
):
    stitched_grid = np.asarray(stitch_sequence[0][0]["grid"], dtype=float).copy()
    stitched_analytic = align_to_value(
        np.asarray(stitch_sequence[0][0]["analytic"], dtype=float),
        stitched_grid,
        anchor_x,
        0.0,
    )
    state_relative_values = (
        {
            str(state): float(value)
            for state, value in (state_relative_values or {}).items()
            if np.isfinite(float(value))
        }
        if state_relative_values
        else {}
    )
    state_boot_relative_values = (
        {
            str(state): np.asarray(values, dtype=float)
            for state, values in (state_boot_relative_values or {}).items()
        }
        if state_boot_relative_values
        else {}
    )

    segment_contributions = []
    anchor_markers = []
    current_anchor_value = 0.0
    state_shift = None
    for record, orientation in stitch_sequence:
        if orientation == "forward":
            start_anchor = float(record["source_x_most"])
            end_anchor = float(record["target_x_most"])
            start_state = str(record["source_name"])
            end_state = str(record["target_name"])
        else:
            start_anchor = float(record["target_x_most"])
            end_anchor = float(record["source_x_most"])
            start_state = str(record["target_name"])
            end_state = str(record["source_name"])
        anchor_markers.extend([start_anchor, end_anchor])
        start_target_value = current_anchor_value
        if start_state in state_relative_values:
            if state_shift is None:
                state_shift = current_anchor_value - float(state_relative_values[start_state])
            start_target_value = state_shift + float(state_relative_values[start_state])
        aligned = align_to_value(
            np.asarray(record["mts_pmf_ref0"], dtype=float),
            record["grid"],
            start_anchor,
            start_target_value,
        )
        interval_piece = masked_interval(aligned, record["grid"], start_anchor, end_anchor)
        segment_contributions.append(interval_piece)
        end_value = value_at_x(interval_piece, record["grid"], end_anchor)
        if state_shift is not None and end_state in state_relative_values:
            current_anchor_value = state_shift + float(state_relative_values[end_state])
        elif np.isfinite(end_value):
            current_anchor_value = end_value

    stitched_mean, _, stitched_counts = combine_contributions(segment_contributions)
    anchor_markers = list(dict.fromkeys(round(float(x), 8) for x in anchor_markers))

    segment_boot_stacks = [
        np.asarray(record["mts_boot_stack"], dtype=float)
        for record, _ in stitch_sequence
    ]
    boot_counts = [int(stack.shape[0]) for stack in segment_boot_stacks]
    stitched_boot_curves = []
    if boot_counts and all(count > 0 for count in boot_counts):
        n_boot_common = min(boot_counts)
        if state_boot_relative_values:
            state_boot_counts = [
                int(np.asarray(values, dtype=float).shape[0])
                for values in state_boot_relative_values.values()
            ]
            if state_boot_counts:
                n_boot_common = min(n_boot_common, min(state_boot_counts))
        for boot_idx in range(n_boot_common):
            boot_anchor_value = 0.0
            boot_state_shift = None
            boot_state_values = {
                state: float(values[boot_idx])
                for state, values in state_boot_relative_values.items()
                if int(np.asarray(values, dtype=float).shape[0]) > boot_idx
                and np.isfinite(values[boot_idx])
            }
            boot_contribs = []
            for (record, orientation), boot_stack in zip(stitch_sequence, segment_boot_stacks):
                if orientation == "forward":
                    start_anchor = float(record["source_x_most"])
                    end_anchor = float(record["target_x_most"])
                    start_state = str(record["source_name"])
                    end_state = str(record["target_name"])
                else:
                    start_anchor = float(record["target_x_most"])
                    end_anchor = float(record["source_x_most"])
                    start_state = str(record["target_name"])
                    end_state = str(record["source_name"])
                sample = boot_stack[boot_idx % boot_stack.shape[0]].copy()
                start_target_value = boot_anchor_value
                if start_state in boot_state_values:
                    if boot_state_shift is None:
                        boot_state_shift = boot_anchor_value - float(boot_state_values[start_state])
                    start_target_value = boot_state_shift + float(boot_state_values[start_state])
                sample = align_to_value(sample, record["grid"], start_anchor, start_target_value)
                interval_piece = masked_interval(sample, record["grid"], start_anchor, end_anchor)
                boot_contribs.append(interval_piece)
                end_value = value_at_x(interval_piece, record["grid"], end_anchor)
                if boot_state_shift is not None and end_state in boot_state_values:
                    boot_anchor_value = boot_state_shift + float(boot_state_values[end_state])
                elif np.isfinite(end_value):
                    boot_anchor_value = end_value
            boot_mean, _, _ = combine_contributions(boot_contribs)
            stitched_boot_curves.append(boot_mean)
    if stitched_boot_curves:
        boot_stack = np.vstack(stitched_boot_curves)
        with np.errstate(invalid="ignore"):
            boot_mean = np.nanmean(boot_stack, axis=0)
            boot_var = np.nanvar(boot_stack, axis=0, ddof=1 if boot_stack.shape[0] > 1 else 0)
    else:
        boot_stack = np.empty((0, len(stitched_grid)), dtype=float)
        boot_mean = np.full(len(stitched_grid), np.nan, dtype=float)
        boot_var = np.full(len(stitched_grid), np.nan, dtype=float)

    return {
        "anchor": anchor_name,
        "anchor_x": float(anchor_x),
        "stitched_grid": stitched_grid,
        "stitched_analytic": stitched_analytic,
        "stitched_mean": stitched_mean,
        "stitched_counts": stitched_counts,
        "anchor_markers": anchor_markers,
        "boot_stack": boot_stack,
        "boot_mean": boot_mean,
        "boot_var": boot_var,
    }


def pair_js_divergence(source_eq_x, target_eq_x, grid):
    source_eq_x = np.asarray(source_eq_x, dtype=float)
    target_eq_x = np.asarray(target_eq_x, dtype=float)
    source_eq_x = source_eq_x[np.isfinite(source_eq_x)]
    target_eq_x = target_eq_x[np.isfinite(target_eq_x)]
    grid = np.asarray(grid, dtype=float)
    if source_eq_x.size == 0 or target_eq_x.size == 0 or grid.size == 0:
        return float("nan")
    edges = build_edges_from_grid(grid)
    source_hist, _ = np.histogram(source_eq_x, bins=edges, density=False)
    target_hist, _ = np.histogram(target_eq_x, bins=edges, density=False)
    source_prob = source_hist.astype(float)
    target_prob = target_hist.astype(float)
    if float(np.sum(source_prob)) <= 0.0 or float(np.sum(target_prob)) <= 0.0:
        return float("nan")
    source_prob /= float(np.sum(source_prob))
    target_prob /= float(np.sum(target_prob))
    midpoint = 0.5 * (source_prob + target_prob)
    mask_source = source_prob > 0.0
    mask_target = target_prob > 0.0
    kl_source = float(np.sum(source_prob[mask_source] * np.log(source_prob[mask_source] / midpoint[mask_source])))
    kl_target = float(np.sum(target_prob[mask_target] * np.log(target_prob[mask_target] / midpoint[mask_target])))
    return float(0.5 * (kl_source + kl_target))


def _shift_curve(values, shift):
    values = np.asarray(values, dtype=float).copy()
    finite_mask = np.isfinite(values)
    values[finite_mask] += float(shift)
    return values


def _shift_stack(stack, shifts):
    stack = np.asarray(stack, dtype=float).copy()
    if stack.ndim != 2 or stack.shape[0] == 0:
        return np.empty((0, stack.shape[1] if stack.ndim == 2 else 0), dtype=float)
    shifts = np.asarray(shifts, dtype=float).reshape((-1, 1))
    finite_mask = np.isfinite(stack)
    out = stack.copy()
    out[finite_mask] += np.broadcast_to(shifts, out.shape)[finite_mask]
    return out


def _stack_quantiles(stack):
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 2 or stack.shape[0] == 0:
        n_cols = int(stack.shape[1]) if stack.ndim == 2 else 0
        nan = np.full(n_cols, np.nan, dtype=float)
        return nan.copy(), nan.copy(), nan.copy(), nan.copy()
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        q05 = np.nanpercentile(stack, 5.0, axis=0)
        q95 = np.nanpercentile(stack, 95.0, axis=0)
        var = np.nanvar(stack, axis=0, ddof=1 if stack.shape[0] > 1 else 0)
    return mean, q05, q95, var


def _expand_segment_values(mask, values, total_len):
    out = np.full(int(total_len), np.nan, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    values = np.asarray(values, dtype=float)
    if values.size == int(np.sum(mask)):
        out[mask] = values
    return out


def _expand_segment_stack(mask, stack, total_len):
    mask = np.asarray(mask, dtype=bool)
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 2 or stack.shape[0] == 0:
        return np.empty((0, int(total_len)), dtype=float)
    out = np.full((stack.shape[0], int(total_len)), np.nan, dtype=float)
    if stack.shape[1] == int(np.sum(mask)):
        out[:, mask] = stack
    return out


def _ordered_window_names_from_stitch_sequence(stitch_sequence):
    if not stitch_sequence:
        return []
    names = []
    previous_end = None
    for record, orientation in stitch_sequence:
        if orientation == "forward":
            start_name = str(record["source_name"])
            end_name = str(record["target_name"])
        else:
            start_name = str(record["target_name"])
            end_name = str(record["source_name"])
        if not names:
            names.append(start_name)
        elif previous_end is not None and start_name != previous_end:
            names.append(start_name)
        if end_name != names[-1]:
            names.append(end_name)
        previous_end = end_name
    return names


def _bootstrap_raw_eq_stack(window_rows, grid, rng_seed, run_context, n_boot):
    analytic_ref = background_potential_1d(grid, run_context)
    pmf_raw, ess, probability = direct_eq_mbar_pmf(window_rows, grid, run_context)
    rng = np.random.default_rng(int(rng_seed))
    boot_curves = []
    for _ in range(int(n_boot)):
        boot_rows = []
        for row in window_rows:
            samples = np.asarray(row["tail_x"], dtype=float)
            samples = samples[np.isfinite(samples)]
            if samples.size <= 0:
                continue
            boot_rows.append(
                {
                    "tail_x": rng.choice(samples, size=samples.size, replace=True),
                    "x_m": float(row["x_m"]),
                    "k_m": float(row["k_m"]),
                }
            )
        boot_pmf, _, _ = direct_eq_mbar_pmf(boot_rows, grid, run_context)
        boot_curves.append(np.asarray(boot_pmf, dtype=float))
    boot_stack = (
        np.vstack(boot_curves)
        if boot_curves
        else np.empty((0, len(grid)), dtype=float)
    )
    return {
        "analytic": analytic_ref,
        "pmf_raw": np.asarray(pmf_raw, dtype=float),
        "boot_stack_raw": boot_stack,
        "ess": np.asarray(ess, dtype=float),
        "probability": np.asarray(probability, dtype=float),
    }


def build_eq_cluster_component(
    window_rows,
    grid,
    run_context,
    n_boot,
    rng_seed,
    cluster_index,
):
    grid = np.asarray(grid, dtype=float)
    source_name = str(window_rows[0]["name"])
    target_name = str(window_rows[-1]["name"])
    source_x = float(window_rows[0]["x_most"])
    target_x = float(window_rows[-1]["x_most"])
    raw_payload = _bootstrap_raw_eq_stack(
        window_rows,
        grid,
        rng_seed,
        run_context,
        n_boot,
    )
    pmf_raw = np.asarray(raw_payload["pmf_raw"], dtype=float)
    boot_raw = np.asarray(raw_payload["boot_stack_raw"], dtype=float)
    pmf_source_zero = align_to_zero_at_anchor(pmf_raw, grid, source_x)
    pmf_target_zero = align_to_zero_at_anchor(pmf_raw, grid, target_x)
    delta_base = value_at_x(pmf_source_zero, grid, target_x)
    pmf_target_in_source = _shift_curve(pmf_target_zero, delta_base)

    if boot_raw.ndim == 1:
        boot_raw = boot_raw[None, :]
    if boot_raw.size > 0:
        boot_source_zero = np.vstack(
            [align_to_zero_at_anchor(sample, grid, source_x) for sample in boot_raw]
        )
        boot_target_zero = np.vstack(
            [align_to_zero_at_anchor(sample, grid, target_x) for sample in boot_raw]
        )
        delta_boot = np.asarray(
            [value_at_x(sample, grid, target_x) for sample in boot_source_zero],
            dtype=float,
        )
        boot_target_in_source = _shift_stack(boot_target_zero, delta_boot)
        _, _, _, var_source_zero = _stack_quantiles(boot_source_zero)
        _, _, _, var_target_zero = _stack_quantiles(boot_target_zero)
    else:
        boot_source_zero = np.empty((0, len(grid)), dtype=float)
        boot_target_zero = np.empty((0, len(grid)), dtype=float)
        boot_target_in_source = np.empty((0, len(grid)), dtype=float)
        delta_boot = np.empty((0,), dtype=float)
        var_source_zero = np.full(len(grid), np.nan, dtype=float)
        var_target_zero = np.full(len(grid), np.nan, dtype=float)

    use_source_mask = np.isfinite(var_source_zero) & (
        ~np.isfinite(var_target_zero) | (var_source_zero <= var_target_zero)
    )
    selected_variance = np.fmin(var_source_zero, var_target_zero)
    selected_pmf_local = np.where(use_source_mask, pmf_source_zero, pmf_target_in_source)
    if boot_source_zero.shape[0] > 0 and boot_target_in_source.shape[0] > 0:
        selected_boot_stack = np.where(
            use_source_mask[None, :],
            boot_source_zero,
            boot_target_in_source,
        )
    else:
        selected_boot_stack = np.empty((0, len(grid)), dtype=float)

    coverage_samples = np.concatenate(
        [
            np.asarray(row["tail_x"], dtype=float)[np.isfinite(np.asarray(row["tail_x"], dtype=float))]
            for row in window_rows
        ]
    )
    coverage_mask = coverage_mask_from_samples(coverage_samples, grid)
    for values in (
        pmf_source_zero,
        pmf_target_zero,
        pmf_target_in_source,
        selected_pmf_local,
        var_source_zero,
        var_target_zero,
        selected_variance,
    ):
        values[~coverage_mask] = np.nan
    if boot_source_zero.size > 0:
        boot_source_zero[:, ~coverage_mask] = np.nan
    if boot_target_zero.size > 0:
        boot_target_zero[:, ~coverage_mask] = np.nan
    if boot_target_in_source.size > 0:
        boot_target_in_source[:, ~coverage_mask] = np.nan
    if selected_boot_stack.size > 0:
        selected_boot_stack[:, ~coverage_mask] = np.nan

    _, boot_low, boot_high, _ = _stack_quantiles(selected_boot_stack)
    return {
        "component_kind": "eq_cluster",
        "pair_kind": "eq_cluster",
        "cluster_index": int(cluster_index),
        "window_names": [str(row["name"]) for row in window_rows],
        "label": " + ".join(str(row["name"]) for row in window_rows),
        "description": "EQ cluster MBAR: " + " + ".join(str(row["name"]) for row in window_rows),
        "source_name": source_name,
        "target_name": target_name,
        "source_x_most": source_x,
        "target_x_most": target_x,
        "grid": grid.copy(),
        "analytic": np.asarray(raw_payload["analytic"], dtype=float).copy(),
        "ess": np.asarray(raw_payload["ess"], dtype=float).copy(),
        "probability": np.asarray(raw_payload["probability"], dtype=float).copy(),
        "pmf_source_zero": pmf_source_zero.copy(),
        "pmf_target_zero": pmf_target_zero.copy(),
        "pmf_target_in_source_frame": pmf_target_in_source.copy(),
        "boot_stack_source_zero": boot_source_zero.copy(),
        "boot_stack_target_zero": boot_target_zero.copy(),
        "boot_stack_target_in_source_frame": boot_target_in_source.copy(),
        "var_source_zero": var_source_zero.copy(),
        "var_target_zero": var_target_zero.copy(),
        "selected_variance": selected_variance.copy(),
        "selected_pmf_local": selected_pmf_local.copy(),
        "selected_boot_stack_local": selected_boot_stack.copy(),
        "selected_boot_low_local": np.asarray(boot_low, dtype=float).copy(),
        "selected_boot_high_local": np.asarray(boot_high, dtype=float).copy(),
        "component_mask": coverage_mask.copy(),
        "mts_pmf_ref0": selected_pmf_local.copy(),
        "mts_boot_stack": selected_boot_stack.copy(),
        "delta_f": value_at_x(selected_pmf_local, grid, target_x),
        "var_dF": value_at_x(selected_variance, grid, target_x),
        "neq_stage_cost": 0.0,
        "neq_step_max": 0.0,
        "source_eq_x": coverage_samples.copy(),
        "target_eq_x": coverage_samples.copy(),
    }


def build_eq_neq_bridge_component(
    bridge_record,
    left_cluster_component,
    right_cluster_component,
    grid,
    n_boot,
):
    grid = np.asarray(grid, dtype=float)
    source_x = float(left_cluster_component["target_x_most"])
    target_x = float(right_cluster_component["source_x_most"])
    interval_mask = interval_coverage_mask(grid, source_x, target_x)
    bridge_offset_base = float(
        bridge_record.get(
            "delta_f",
            value_at_x(np.asarray(bridge_record["mts_pmf_ref0"], dtype=float), grid, target_x),
        )
    )
    left_eq_pmf = np.asarray(left_cluster_component["pmf_target_zero"], dtype=float).copy()
    right_eq_pmf = _shift_curve(
        np.asarray(right_cluster_component["pmf_source_zero"], dtype=float),
        bridge_offset_base,
    )
    left_eq_var = np.asarray(left_cluster_component["var_target_zero"], dtype=float).copy()
    right_eq_var = np.asarray(right_cluster_component["var_source_zero"], dtype=float).copy()

    left_eq_stack = np.asarray(left_cluster_component["boot_stack_target_zero"], dtype=float)
    right_eq_stack_source = np.asarray(right_cluster_component["boot_stack_source_zero"], dtype=float)

    bridge_profile = fragment_variance_profile(
        bridge_record,
        source_x,
        target_x,
        grid,
        n_boot,
    )
    segment_mask = np.asarray(bridge_profile["mask"], dtype=bool)
    neq_forward_pmf = _expand_segment_values(
        segment_mask,
        bridge_profile["pmf_forward_source_zero"],
        len(grid),
    )
    neq_reverse_pmf_target = _expand_segment_values(
        segment_mask,
        bridge_profile["pmf_reverse_target_zero"],
        len(grid),
    )
    neq_forward_var = _expand_segment_values(
        segment_mask,
        bridge_profile["var_forward_source_zero"],
        len(grid),
    )
    neq_reverse_var = _expand_segment_values(
        segment_mask,
        bridge_profile["var_reverse_target_zero"],
        len(grid),
    )
    neq_forward_stack = _expand_segment_stack(
        segment_mask,
        bridge_profile["boot_stack_forward_source_zero"],
        len(grid),
    )
    neq_reverse_stack_target = _expand_segment_stack(
        segment_mask,
        bridge_profile["boot_stack_reverse_target_zero"],
        len(grid),
    )
    bridge_delta_base = value_at_x(neq_forward_pmf, grid, target_x)
    if not np.isfinite(bridge_delta_base):
        bridge_delta_base = bridge_offset_base
    neq_reverse_pmf = _shift_curve(neq_reverse_pmf_target, bridge_delta_base)

    if neq_forward_stack.shape[0] > 0:
        bridge_delta_boot = np.asarray(
            [value_at_x(sample, grid, target_x) for sample in neq_forward_stack],
            dtype=float,
        )
        neq_reverse_stack = _shift_stack(neq_reverse_stack_target, bridge_delta_boot)
    else:
        bridge_delta_boot = np.empty((0,), dtype=float)
        neq_reverse_stack = np.empty((0, len(grid)), dtype=float)

    eq_use_left_mask = np.isfinite(left_eq_var) & (
        ~np.isfinite(right_eq_var) | (left_eq_var <= right_eq_var)
    )
    eq_var = np.fmin(left_eq_var, right_eq_var)
    eq_pmf_local = np.where(eq_use_left_mask, left_eq_pmf, right_eq_pmf)

    neq_use_forward_mask = np.isfinite(neq_forward_var) & (
        ~np.isfinite(neq_reverse_var) | (neq_forward_var <= neq_reverse_var)
    )
    neq_var = np.fmin(neq_forward_var, neq_reverse_var)
    neq_pmf_local = np.where(neq_use_forward_mask, neq_forward_pmf, neq_reverse_pmf)

    preferred_mask = (
        interval_mask
        & np.isfinite(eq_var)
        & np.isfinite(neq_var)
        & (neq_var < eq_var)
    )
    neq_regime_mask = np.zeros(len(grid), dtype=bool)
    if np.any(preferred_mask):
        preferred_indices = np.where(preferred_mask)[0]
        neq_regime_mask[preferred_indices[0] : preferred_indices[-1] + 1] = True
        neq_regime_mask &= interval_mask

    selected_pmf_local = eq_pmf_local.copy()
    selected_pmf_local[neq_regime_mask] = neq_pmf_local[neq_regime_mask]
    selected_variance = eq_var.copy()
    selected_variance[neq_regime_mask] = neq_var[neq_regime_mask]
    selected_pmf_local[~interval_mask] = np.nan
    selected_variance[~interval_mask] = np.nan

    positive_boot_counts = [
        int(stack.shape[0])
        for stack in (left_eq_stack, right_eq_stack_source, neq_forward_stack, neq_reverse_stack)
        if stack.ndim == 2 and stack.shape[0] > 0
    ]
    if positive_boot_counts:
        n_boot_common = min(positive_boot_counts)
        left_eq_stack = left_eq_stack[:n_boot_common].copy()
        right_eq_stack = _shift_stack(
            right_eq_stack_source[:n_boot_common],
            bridge_delta_boot[:n_boot_common] if bridge_delta_boot.size >= n_boot_common else np.full(n_boot_common, bridge_delta_base, dtype=float),
        )
        neq_forward_stack = neq_forward_stack[:n_boot_common].copy()
        neq_reverse_stack = neq_reverse_stack[:n_boot_common].copy()
        eq_boot_stack = np.where(eq_use_left_mask[None, :], left_eq_stack, right_eq_stack)
        neq_boot_stack = np.where(neq_use_forward_mask[None, :], neq_forward_stack, neq_reverse_stack)
        selected_boot_stack = eq_boot_stack.copy()
        selected_boot_stack[:, neq_regime_mask] = neq_boot_stack[:, neq_regime_mask]
        selected_boot_stack[:, ~interval_mask] = np.nan
    else:
        selected_boot_stack = np.empty((0, len(grid)), dtype=float)

    _, boot_low, boot_high, _ = _stack_quantiles(selected_boot_stack)
    return {
        "component_kind": "eq_neq_bridge",
        "pair_kind": "eq_neq_bridge",
        "label": f"{left_cluster_component['target_name']} -> {right_cluster_component['source_name']}",
        "description": f"EQ-cluster / NEQ bridge hybrid: {left_cluster_component['target_name']} -> {right_cluster_component['source_name']}",
        "source_name": str(left_cluster_component["target_name"]),
        "target_name": str(right_cluster_component["source_name"]),
        "source_x_most": source_x,
        "target_x_most": target_x,
        "grid": grid.copy(),
        "analytic": np.asarray(bridge_record["analytic"], dtype=float).copy(),
        "selected_pmf_local": selected_pmf_local.copy(),
        "selected_variance": selected_variance.copy(),
        "selected_boot_stack_local": selected_boot_stack.copy(),
        "selected_boot_low_local": np.asarray(boot_low, dtype=float).copy(),
        "selected_boot_high_local": np.asarray(boot_high, dtype=float).copy(),
        "component_mask": interval_mask.copy(),
        "mts_pmf_ref0": selected_pmf_local.copy(),
        "mts_boot_stack": selected_boot_stack.copy(),
        "pmf_eq_local": eq_pmf_local.copy(),
        "var_eq": eq_var.copy(),
        "pmf_neq_local": neq_pmf_local.copy(),
        "var_neq": neq_var.copy(),
        "neq_regime_mask": neq_regime_mask.copy(),
        "eq_use_left_mask": eq_use_left_mask.copy(),
        "neq_use_forward_mask": neq_use_forward_mask.copy(),
        "delta_f": value_at_x(selected_pmf_local, grid, target_x),
        "var_dF": value_at_x(selected_variance, grid, target_x),
        "neq_stage_cost": float(bridge_record.get("neq_stage_cost", bridge_record.get("neq_step_max", 0.0))),
        "neq_step_max": float(bridge_record.get("neq_step_max", bridge_record.get("neq_stage_cost", 0.0))),
        "bridge_segment_dir": str(bridge_record.get("segment_dir", "")),
        "bridge_pair_kind": str(bridge_record.get("pair_kind", "")),
        "left_cluster_windows": list(left_cluster_component["window_names"]),
        "right_cluster_windows": list(right_cluster_component["window_names"]),
    }


def build_eq_cluster_bridge_bundle(
    raw_stitch_sequence,
    window_payloads,
    js_threshold,
    run_context,
    n_boot,
    rng_seed,
):
    raw_stitch_sequence = list(raw_stitch_sequence or [])
    window_payloads = list(window_payloads or [])
    if not raw_stitch_sequence or not window_payloads:
        return {
            "component_sequence": [],
            "component_records": [],
            "cluster_records": [],
            "bridge_records": [],
            "adjacency_rows": [],
            "bundle": {
                "anchor": "",
                "anchor_x": float("nan"),
                "stitched_grid": np.asarray([], dtype=float),
                "stitched_analytic": np.asarray([], dtype=float),
                "stitched_mean": np.asarray([], dtype=float),
                "stitched_counts": np.asarray([], dtype=float),
                "anchor_markers": [],
                "boot_stack": np.empty((0, 0), dtype=float),
                "boot_mean": np.asarray([], dtype=float),
                "boot_var": np.asarray([], dtype=float),
            },
        }

    grid = np.asarray(raw_stitch_sequence[0][0]["grid"], dtype=float)
    window_lookup = {str(row["name"]): row for row in window_payloads}
    ordered_window_names = _ordered_window_names_from_stitch_sequence(raw_stitch_sequence)
    adjacency_rows = []
    clusters = []
    if ordered_window_names:
        current_cluster = [ordered_window_names[0]]
        for left_name, right_name in zip(ordered_window_names[:-1], ordered_window_names[1:]):
            left_window = window_lookup.get(str(left_name))
            right_window = window_lookup.get(str(right_name))
            pair_js = float("nan")
            if left_window is not None and right_window is not None:
                pair_js = pair_js_divergence(left_window["tail_x"], right_window["tail_x"], grid)
            adjacency_rows.append(
                {
                    "left_name": str(left_name),
                    "right_name": str(right_name),
                    "pair_jsd": float(pair_js),
                    "threshold": float(js_threshold),
                    "same_cluster": bool(np.isfinite(pair_js) and pair_js <= float(js_threshold)),
                }
            )
            if np.isfinite(pair_js) and pair_js <= float(js_threshold):
                current_cluster.append(str(right_name))
            else:
                clusters.append(current_cluster)
                current_cluster = [str(right_name)]
        clusters.append(current_cluster)

    component_records = []
    cluster_records = []
    bridge_records = []
    bridge_lookup = {}
    for record, orientation in raw_stitch_sequence:
        if orientation == "forward":
            bridge_lookup[(str(record["source_name"]), str(record["target_name"]))] = record
        else:
            bridge_lookup[(str(record["target_name"]), str(record["source_name"]))] = record

    for cluster_index, cluster_names in enumerate(clusters):
        cluster_rows = [window_lookup[name] for name in cluster_names if name in window_lookup]
        if not cluster_rows:
            continue
        cluster_component = build_eq_cluster_component(
            cluster_rows,
            grid,
            run_context,
            n_boot,
            int(rng_seed) + 1000 * int(cluster_index),
            cluster_index,
        )
        cluster_records.append(cluster_component)
    if not cluster_records:
        return {
            "component_sequence": [],
            "component_records": [],
            "cluster_records": [],
            "bridge_records": [],
            "adjacency_rows": adjacency_rows,
            "bundle": {
                "anchor": "",
                "anchor_x": float("nan"),
                "stitched_grid": grid.copy(),
                "stitched_analytic": np.full(len(grid), np.nan, dtype=float),
                "stitched_mean": np.full(len(grid), np.nan, dtype=float),
                "stitched_counts": np.zeros(len(grid), dtype=float),
                "anchor_markers": [],
                "boot_stack": np.empty((0, len(grid)), dtype=float),
                "boot_mean": np.full(len(grid), np.nan, dtype=float),
                "boot_var": np.full(len(grid), np.nan, dtype=float),
            },
        }

    component_records.append(cluster_records[0])
    for left_cluster, right_cluster in zip(cluster_records[:-1], cluster_records[1:]):
        bridge_record = bridge_lookup.get((str(left_cluster["target_name"]), str(right_cluster["source_name"])))
        if bridge_record is not None:
            bridge_component = build_eq_neq_bridge_component(
                bridge_record,
                left_cluster,
                right_cluster,
                grid,
                n_boot,
            )
            bridge_records.append(bridge_component)
            component_records.append(bridge_component)
        component_records.append(right_cluster)

    positive_boot_counts = [
        int(np.asarray(row["selected_boot_stack_local"], dtype=float).shape[0])
        for row in component_records
        if np.asarray(row["selected_boot_stack_local"], dtype=float).ndim == 2
        and int(np.asarray(row["selected_boot_stack_local"], dtype=float).shape[0]) > 0
    ]
    n_boot_common = min(positive_boot_counts) if positive_boot_counts else 0
    offset = 0.0
    boot_offsets = np.zeros(n_boot_common, dtype=float)
    anchor_markers = []
    contributor_counts = []
    analytic_ref = align_to_zero_at_anchor(
        background_potential_1d(grid, run_context),
        grid,
        float(cluster_records[0]["source_x_most"]),
    )
    for row in component_records:
        local_mean = np.asarray(row["selected_pmf_local"], dtype=float)
        local_stack = np.asarray(row["selected_boot_stack_local"], dtype=float)
        if local_stack.ndim == 1:
            local_stack = local_stack[None, :]
        global_mean = _shift_curve(local_mean, offset)
        if n_boot_common > 0 and local_stack.shape[0] >= n_boot_common:
            global_stack = _shift_stack(local_stack[:n_boot_common], boot_offsets)
        else:
            global_stack = np.empty((0, len(grid)), dtype=float)
        _, global_low, global_high, global_var = _stack_quantiles(global_stack)
        row["global_selected_pmf"] = global_mean
        row["global_selected_boot_stack"] = global_stack
        row["global_selected_boot_low"] = global_low
        row["global_selected_boot_high"] = global_high
        row["global_selected_variance"] = (
            global_var if global_stack.shape[0] > 0 else np.asarray(row["selected_variance"], dtype=float).copy()
        )
        contributor_counts.append(np.isfinite(global_mean))
        anchor_markers.extend([float(row["source_x_most"]), float(row["target_x_most"])])
        next_offset = value_at_x(global_mean, grid, float(row["target_x_most"]))
        if np.isfinite(next_offset):
            offset = float(next_offset)
        if n_boot_common > 0 and global_stack.shape[0] >= n_boot_common:
            target_idx = int(np.argmin(np.abs(grid - float(row["target_x_most"]))))
            next_boot_offsets = np.asarray(global_stack[:, target_idx], dtype=float)
            next_boot_offsets[~np.isfinite(next_boot_offsets)] = float(offset)
            boot_offsets = next_boot_offsets

    if contributor_counts:
        stitched_counts = np.sum(np.vstack(contributor_counts), axis=0).astype(float)
    else:
        stitched_counts = np.zeros(len(grid), dtype=float)

    stitched_mean = np.full(len(grid), np.nan, dtype=float)
    stitched_var = np.full(len(grid), np.nan, dtype=float)
    selection_index = np.full(len(grid), -1, dtype=int)
    for grid_idx in range(len(grid)):
        best_idx = None
        best_var = float("inf")
        for component_idx, row in enumerate(component_records):
            value = float(np.asarray(row["global_selected_pmf"], dtype=float)[grid_idx])
            variance = float(np.asarray(row["global_selected_variance"], dtype=float)[grid_idx])
            if not np.isfinite(value):
                continue
            if np.isfinite(variance) and variance < best_var:
                best_var = variance
                best_idx = component_idx
            elif best_idx is None:
                best_idx = component_idx
        if best_idx is None:
            continue
        selection_index[grid_idx] = int(best_idx)
        stitched_mean[grid_idx] = float(
            np.asarray(component_records[best_idx]["global_selected_pmf"], dtype=float)[grid_idx]
        )
        stitched_var[grid_idx] = float(
            np.asarray(component_records[best_idx]["global_selected_variance"], dtype=float)[grid_idx]
        )

    if n_boot_common > 0:
        stitched_boot_stack = np.full((n_boot_common, len(grid)), np.nan, dtype=float)
        for grid_idx, component_idx in enumerate(selection_index):
            if int(component_idx) < 0:
                continue
            component_stack = np.asarray(
                component_records[int(component_idx)]["global_selected_boot_stack"],
                dtype=float,
            )
            if component_stack.ndim == 2 and component_stack.shape[0] >= n_boot_common:
                stitched_boot_stack[:, grid_idx] = component_stack[:n_boot_common, grid_idx]
        with np.errstate(invalid="ignore"):
            stitched_boot_mean = np.nanmean(stitched_boot_stack, axis=0)
            stitched_boot_var = np.nanvar(
                stitched_boot_stack,
                axis=0,
                ddof=1 if n_boot_common > 1 else 0,
            )
    else:
        stitched_boot_stack = np.empty((0, len(grid)), dtype=float)
        stitched_boot_mean = np.full(len(grid), np.nan, dtype=float)
        stitched_boot_var = stitched_var.copy()

    for grid_idx, component_idx in enumerate(selection_index):
        if int(component_idx) >= 0:
            component_records[int(component_idx)].setdefault("selected_grid_indices", []).append(int(grid_idx))

    bundle = {
        "anchor": str(cluster_records[0]["source_name"]),
        "anchor_x": float(cluster_records[0]["source_x_most"]),
        "stitched_grid": grid.copy(),
        "stitched_analytic": analytic_ref.copy(),
        "stitched_mean": stitched_mean.copy(),
        "stitched_counts": stitched_counts.copy(),
        "anchor_markers": list(dict.fromkeys(round(float(x), 8) for x in anchor_markers)),
        "boot_stack": stitched_boot_stack.copy(),
        "boot_mean": np.asarray(stitched_boot_mean, dtype=float).copy(),
        "boot_var": np.asarray(stitched_boot_var, dtype=float).copy(),
        "selection_index": selection_index.copy(),
    }
    component_sequence = [(row, "forward") for row in component_records]
    return {
        "component_sequence": component_sequence,
        "component_records": component_records,
        "cluster_records": cluster_records,
        "bridge_records": bridge_records,
        "adjacency_rows": adjacency_rows,
        "bundle": bundle,
    }


def build_rescue_stage_plot_data(
    stop_reason,
    rescue_steps,
    segment_pair_record_by_dir,
    summary_seed,
    n_boot,
    window_payloads,
    run_context,
    js_threshold,
    left_sequence=None,
    right_sequence=None,
    base_pair_record=None,
    base_eq_overlap_pair_record=None,
    base_stitch_sequence=None,
    active_stitch_sequence=None,
    eq_overlap_pair_record=None,
):
    left_sequence = list(left_sequence or [])
    right_sequence = list(right_sequence or [])
    base_stitch_sequence = list(base_stitch_sequence or [])
    active_stitch_sequence = list(active_stitch_sequence or [])
    if stop_reason != "rescue_window":
        return []

    eq_overlap_records_by_summary_file = {
        str(record.get("summary_file", "")): record
        for record, _ in base_stitch_sequence
        if str(record.get("pair_kind", "")) == "eq_overlap_mbar"
        and str(record.get("summary_file", ""))
    }
    for record, _ in active_stitch_sequence:
        if (
            str(record.get("pair_kind", "")) == "eq_overlap_mbar"
            and str(record.get("summary_file", ""))
        ):
            eq_overlap_records_by_summary_file[str(record.get("summary_file", ""))] = record
    if (
        base_eq_overlap_pair_record is not None
        and str(base_eq_overlap_pair_record.get("summary_file", ""))
    ):
        eq_overlap_records_by_summary_file[
            str(base_eq_overlap_pair_record.get("summary_file", ""))
        ] = base_eq_overlap_pair_record
    if eq_overlap_pair_record is not None and str(eq_overlap_pair_record.get("summary_file", "")):
        eq_overlap_records_by_summary_file[str(eq_overlap_pair_record.get("summary_file", ""))] = (
            eq_overlap_pair_record
        )

    def stitch_sequence_from_segment_refs(segment_refs, fallback_eq_overlap_record):
        if not segment_refs:
            return []
        if (
            fallback_eq_overlap_record is not None
            and str(fallback_eq_overlap_record.get("summary_file", ""))
        ):
            eq_overlap_records_by_summary_file[
                str(fallback_eq_overlap_record.get("summary_file", ""))
            ] = fallback_eq_overlap_record
        out = []
        for ref in segment_refs:
            kind = str(ref.get("kind", ""))
            orientation = str(ref.get("orientation", "forward"))
            if kind == "nes_segment":
                record = segment_pair_record_by_dir.get(str(ref.get("segment_dir", "")))
            elif kind == "eq_overlap_mbar":
                record = eq_overlap_records_by_summary_file.get(str(ref.get("summary_file", "")))
            else:
                record = None
            if record is None:
                continue
            out.append((record, orientation))
        return out

    stage_sequences = []
    if base_stitch_sequence:
        stage_sequences.append(
            {
                "stage_index": 0,
                "stage_label": "Before rescue",
                "stitch_sequence": list(base_stitch_sequence),
                "target_x": float("nan"),
                "rescue_index": 0,
            }
        )
        for rescue_order_idx, rescue_step in enumerate(rescue_steps):
            rescue_index = int(rescue_step.get("rescue_index", rescue_order_idx + 1))
            stage_stitch_sequence = stitch_sequence_from_segment_refs(
                rescue_step.get("active_stitched_segment_refs_after_step")
                or rescue_step.get("active_segment_refs_after_step")
                or [],
                base_eq_overlap_pair_record,
            )
            if not stage_stitch_sequence:
                continue
            stage_sequences.append(
                {
                    "stage_index": int(rescue_order_idx + 1),
                    "stage_label": f"After rescue {rescue_index}",
                    "stitch_sequence": stage_stitch_sequence,
                    "target_x": float(rescue_step.get("target_x", float("nan"))),
                    "rescue_index": int(rescue_index),
                }
            )
    else:
        if base_pair_record is None and base_eq_overlap_pair_record is None:
            return []
        active_chain_records = (
            [base_pair_record] if base_pair_record is not None else [base_eq_overlap_pair_record]
        )
        stage_sequences = [
            {
                "stage_index": 0,
                "stage_label": "Before rescue",
                "stitch_sequence": left_sequence
                + [(row, "forward") for row in active_chain_records]
                + right_sequence,
                "target_x": float("nan"),
                "rescue_index": 0,
            }
        ]
        for rescue_order_idx, rescue_step in enumerate(rescue_steps):
            overlap_summary_file = str(rescue_step.get("overlap_summary_file", ""))
            if overlap_summary_file and overlap_summary_file in eq_overlap_records_by_summary_file:
                replacement_records = [eq_overlap_records_by_summary_file[overlap_summary_file]]
            else:
                replacement_records = [
                    segment_pair_record_by_dir[path]
                    for path in (
                        str(rescue_step.get("left_rescue_segment_dir", "")),
                        str(rescue_step.get("right_rescue_segment_dir", "")),
                    )
                    if path in segment_pair_record_by_dir
                ]
            base_segment_dir = str(rescue_step.get("base_segment_dir", ""))
            replace_idx = next(
                (
                    idx
                    for idx, row in enumerate(active_chain_records)
                    if str(row.get("segment_dir", "")) == base_segment_dir
                ),
                None,
            )
            if replace_idx is None:
                active_chain_records.extend(replacement_records)
            else:
                active_chain_records[replace_idx : replace_idx + 1] = replacement_records
            rescue_index = int(rescue_step.get("rescue_index", rescue_order_idx + 1))
            stage_sequences.append(
                {
                    "stage_index": int(rescue_order_idx + 1),
                    "stage_label": f"After rescue {rescue_index}",
                    "stitch_sequence": left_sequence
                    + [(row, "forward") for row in active_chain_records]
                    + right_sequence,
                    "target_x": float(rescue_step.get("target_x", float("nan"))),
                    "rescue_index": int(rescue_index),
                }
            )

    plot_data = []
    for stage in stage_sequences:
        raw_stage_stitch_sequence = list(stage["stitch_sequence"])
        if not raw_stage_stitch_sequence:
            continue
        stage_protocol = build_eq_cluster_bridge_bundle(
            raw_stage_stitch_sequence,
            window_payloads,
            js_threshold,
            run_context,
            n_boot,
            93000 + int(summary_seed) + 100 * int(stage["stage_index"]),
        )
        stage_stitch_sequence = list(stage_protocol["component_sequence"])
        if not stage_stitch_sequence:
            continue
        plot_data.append(
            {
                "stage_index": int(stage["stage_index"]),
                "stage_label": str(stage["stage_label"]),
                "rescue_index": int(stage["rescue_index"]),
                "target_x": float(stage["target_x"]),
                "raw_stitch_sequence": raw_stage_stitch_sequence,
                "stitch_sequence": stage_stitch_sequence,
                "active_chain_records": [record for record, _ in stage_stitch_sequence],
                "rescue_records": [
                    record
                    for record, _ in stage_stitch_sequence
                    if str(record.get("component_kind", "")) == "eq_neq_bridge"
                ],
                "graph": None,
                "bundle": stage_protocol["bundle"],
                "component_records": stage_protocol["component_records"],
                "cluster_records": stage_protocol["cluster_records"],
                "bridge_records": stage_protocol["bridge_records"],
                "adjacency_rows": stage_protocol["adjacency_rows"],
            }
        )
    return plot_data


def build_rescue_variance_stages(rescue_stage_plot_data, n_boot):
    rescue_variance_stages = []
    for stage in rescue_stage_plot_data:
        stage_stitch_sequence = list(stage["stitch_sequence"])
        stage_bundle = stage["bundle"]
        stage_grid = np.asarray(stage_bundle["stitched_grid"], dtype=float)
        stage_variance = np.full(stage_grid.shape, np.nan, dtype=float)
        for record, orientation in stage_stitch_sequence:
            if orientation == "forward":
                start_anchor = float(record["source_x_most"])
                end_anchor = float(record["target_x_most"])
            else:
                start_anchor = float(record["target_x_most"])
                end_anchor = float(record["source_x_most"])
            profile = fragment_variance_profile(
                record,
                start_anchor,
                end_anchor,
                stage_grid,
                n_boot,
            )
            profile_values = np.asarray(profile["var_min_oneway"], dtype=float)
            segment_mask = np.asarray(profile["mask"], dtype=bool)
            if profile_values.size != int(np.sum(segment_mask)):
                continue
            stage_segment_variance = np.full(stage_grid.shape, np.nan, dtype=float)
            stage_segment_variance[segment_mask] = profile_values
            existing_mask = np.isfinite(stage_variance)
            incoming_mask = np.isfinite(stage_segment_variance)
            both_mask = existing_mask & incoming_mask
            stage_variance[~existing_mask & incoming_mask] = stage_segment_variance[
                ~existing_mask & incoming_mask
            ]
            stage_variance[both_mask] = np.minimum(
                stage_variance[both_mask],
                stage_segment_variance[both_mask],
            )
        target_x = float(stage["target_x"])
        stage_pmf = np.asarray(stage_bundle["stitched_mean"], dtype=float)
        stage_analytic = np.asarray(stage_bundle["stitched_analytic"], dtype=float)
        rescue_variance_stages.append(
            {
                "stage_index": int(stage["stage_index"]),
                "stage_label": str(stage["stage_label"]),
                "rescue_index": int(stage["rescue_index"]),
                "grid": stage_grid,
                "pmf": stage_pmf,
                "analytic": stage_analytic,
                "variance": stage_variance,
                "target_x": target_x,
                "target_pmf": value_at_x(stage_pmf, stage_grid, target_x)
                if np.isfinite(target_x)
                else float("nan"),
                "target_analytic": value_at_x(stage_analytic, stage_grid, target_x)
                if np.isfinite(target_x)
                else float("nan"),
                "target_variance": value_at_x(stage_variance, stage_grid, target_x)
                if np.isfinite(target_x)
                else float("nan"),
            }
        )
    return rescue_variance_stages


def fragment_variance_profile(record, start_anchor, end_anchor, reference_grid, n_boot):
    grid = np.asarray(record["grid"], dtype=float)
    reference_grid = np.asarray(reference_grid, dtype=float)
    if len(grid) != len(reference_grid) or not np.allclose(grid, reference_grid, equal_nan=True):
        raise RuntimeError("Fragment grids must match the stitched grid.")

    pmf_ref0 = np.asarray(record["mts_pmf_ref0"], dtype=float)
    segment_mask = interval_coverage_mask(grid, start_anchor, end_anchor) & np.isfinite(pmf_ref0)
    segment_grid = grid[segment_mask]
    if segment_grid.size == 0:
        nan = np.full(segment_grid.shape, np.nan, dtype=float)
        return {
            "grid": segment_grid,
            "mask": segment_mask,
            "pmf_endpoint_mean_zero": nan.copy(),
            "boot_stack_endpoint_mean_zero": np.empty((0, segment_grid.size), dtype=float),
            "boot_mean_endpoint_mean_zero": nan.copy(),
            "boot_q05_endpoint_mean_zero": nan.copy(),
            "boot_q95_endpoint_mean_zero": nan.copy(),
            "var_endpoint_mean_zero": nan.copy(),
            "pmf_forward_source_zero": nan.copy(),
            "pmf_reverse_target_zero": nan.copy(),
            "boot_stack_forward_source_zero": np.empty((0, segment_grid.size), dtype=float),
            "boot_stack_reverse_target_zero": np.empty((0, segment_grid.size), dtype=float),
            "boot_mean_forward_source_zero": nan.copy(),
            "boot_mean_reverse_target_zero": nan.copy(),
            "boot_q05_forward_source_zero": nan.copy(),
            "boot_q95_forward_source_zero": nan.copy(),
            "boot_q05_reverse_target_zero": nan.copy(),
            "boot_q95_reverse_target_zero": nan.copy(),
            "var_forward_source_zero": nan.copy(),
            "var_reverse_target_zero": nan.copy(),
            "var_min_directional": nan.copy(),
            "var_min_oneway": nan.copy(),
            "variance_alignment": "min_bidirectional_source_zero_target_zero",
            "n_boot": 0,
        }

    boot_stack = np.asarray(record["mts_boot_stack"], dtype=float)
    if boot_stack.ndim == 1:
        boot_stack = boot_stack[None, :]
    if boot_stack.size > 0:
        pmf_endpoint_mean_zero = align_pmf_to_endpoint_average_zero(
            pmf_ref0,
            grid,
            start_anchor,
            end_anchor,
        )
        endpoint_mean_zero_stack = []
        for sample in boot_stack:
            endpoint_mean_zero = align_pmf_to_endpoint_average_zero(
                sample,
                grid,
                start_anchor,
                end_anchor,
            )
            endpoint_mean_zero_stack.append(
                masked_interval(endpoint_mean_zero, grid, start_anchor, end_anchor)[segment_mask]
            )
        endpoint_mean_zero_stack = np.vstack(endpoint_mean_zero_stack)
        ddof_endpoint = 1 if endpoint_mean_zero_stack.shape[0] > 1 else 0
        with np.errstate(invalid="ignore"):
            boot_mean_endpoint_mean_zero = np.nanmean(endpoint_mean_zero_stack, axis=0)
            boot_q05_endpoint_mean_zero = np.nanpercentile(endpoint_mean_zero_stack, 5.0, axis=0)
            boot_q95_endpoint_mean_zero = np.nanpercentile(endpoint_mean_zero_stack, 95.0, axis=0)
            var_endpoint_mean_zero = np.nanvar(
                endpoint_mean_zero_stack,
                axis=0,
                ddof=ddof_endpoint,
            )
    else:
        pmf_endpoint_mean_zero = np.full(len(grid), np.nan, dtype=float)
        endpoint_mean_zero_stack = np.empty((0, segment_grid.size), dtype=float)
        boot_mean_endpoint_mean_zero = np.full(segment_grid.shape, np.nan, dtype=float)
        boot_q05_endpoint_mean_zero = np.full(segment_grid.shape, np.nan, dtype=float)
        boot_q95_endpoint_mean_zero = np.full(segment_grid.shape, np.nan, dtype=float)
        var_endpoint_mean_zero = np.full(segment_grid.shape, np.nan, dtype=float)

    source_anchor = float(record["source_x_most"])
    target_anchor = float(record["target_x_most"])
    pmf_forward_source_zero = align_to_zero_at_anchor(
        pmf_ref0,
        grid,
        source_anchor,
    )
    pmf_reverse_target_zero = align_to_zero_at_anchor(
        pmf_ref0,
        grid,
        target_anchor,
    )

    if boot_stack.size > 0:
        forward_source_zero_stack = []
        reverse_target_zero_stack = []
        for sample in boot_stack:
            forward_source_zero = align_to_zero_at_anchor(
                sample,
                grid,
                source_anchor,
            )
            reverse_target_zero = align_to_zero_at_anchor(
                sample,
                grid,
                target_anchor,
            )
            forward_source_zero_stack.append(
                masked_interval(forward_source_zero, grid, start_anchor, end_anchor)[segment_mask]
            )
            reverse_target_zero_stack.append(
                masked_interval(reverse_target_zero, grid, start_anchor, end_anchor)[segment_mask]
            )
        forward_stack_segment = np.vstack(forward_source_zero_stack)
        reverse_stack_segment = np.vstack(reverse_target_zero_stack)

        ddof_forward = 1 if forward_stack_segment.shape[0] > 1 else 0
        ddof_reverse = 1 if reverse_stack_segment.shape[0] > 1 else 0
        with np.errstate(invalid="ignore"):
            boot_mean_forward_source_zero = np.nanmean(forward_stack_segment, axis=0)
            boot_mean_reverse_target_zero = np.nanmean(reverse_stack_segment, axis=0)
            boot_q05_forward_source_zero = np.nanpercentile(forward_stack_segment, 5.0, axis=0)
            boot_q95_forward_source_zero = np.nanpercentile(forward_stack_segment, 95.0, axis=0)
            boot_q05_reverse_target_zero = np.nanpercentile(reverse_stack_segment, 5.0, axis=0)
            boot_q95_reverse_target_zero = np.nanpercentile(reverse_stack_segment, 95.0, axis=0)
            var_forward_source_zero = np.nanvar(
                forward_stack_segment,
                axis=0,
                ddof=ddof_forward,
            )
            var_reverse_target_zero = np.nanvar(
                reverse_stack_segment,
                axis=0,
                ddof=ddof_reverse,
            )
        var_min_directional = np.fmin(var_forward_source_zero, var_reverse_target_zero)
        n_boot_used = int(
            min(
                forward_stack_segment.shape[0] if forward_stack_segment.ndim == 2 else 0,
                reverse_stack_segment.shape[0] if reverse_stack_segment.ndim == 2 else 0,
            )
        )
        return {
            "grid": segment_grid,
            "mask": segment_mask,
            "pmf_endpoint_mean_zero": masked_interval(
                pmf_endpoint_mean_zero,
                grid,
                start_anchor,
                end_anchor,
            )[segment_mask],
            "boot_stack_endpoint_mean_zero": endpoint_mean_zero_stack,
            "boot_mean_endpoint_mean_zero": boot_mean_endpoint_mean_zero,
            "boot_q05_endpoint_mean_zero": boot_q05_endpoint_mean_zero,
            "boot_q95_endpoint_mean_zero": boot_q95_endpoint_mean_zero,
            "var_endpoint_mean_zero": var_endpoint_mean_zero,
            "pmf_forward_source_zero": masked_interval(
                pmf_forward_source_zero,
                grid,
                start_anchor,
                end_anchor,
            )[segment_mask],
            "pmf_reverse_target_zero": masked_interval(
                pmf_reverse_target_zero,
                grid,
                start_anchor,
                end_anchor,
            )[segment_mask],
            "boot_stack_forward_source_zero": forward_stack_segment,
            "boot_stack_reverse_target_zero": reverse_stack_segment,
            "boot_mean_forward_source_zero": boot_mean_forward_source_zero,
            "boot_mean_reverse_target_zero": boot_mean_reverse_target_zero,
            "boot_q05_forward_source_zero": boot_q05_forward_source_zero,
            "boot_q95_forward_source_zero": boot_q95_forward_source_zero,
            "boot_q05_reverse_target_zero": boot_q05_reverse_target_zero,
            "boot_q95_reverse_target_zero": boot_q95_reverse_target_zero,
            "var_forward_source_zero": var_forward_source_zero,
            "var_reverse_target_zero": var_reverse_target_zero,
            "var_min_directional": var_min_directional,
            "var_min_oneway": var_min_directional,
            "variance_alignment": "min_bidirectional_source_zero_target_zero",
            "n_boot": n_boot_used,
        }

    boot_stack = np.asarray(record["mts_boot_stack"], dtype=float)
    if boot_stack.ndim == 1:
        boot_stack = boot_stack[None, :]
    if boot_stack.size == 0:
        nan = np.full(segment_grid.shape, np.nan, dtype=float)
        return {
            "grid": segment_grid,
            "mask": segment_mask,
            "pmf_endpoint_mean_zero": nan.copy(),
            "boot_stack_endpoint_mean_zero": np.empty((0, segment_grid.size), dtype=float),
            "boot_mean_endpoint_mean_zero": nan.copy(),
            "boot_q05_endpoint_mean_zero": nan.copy(),
            "boot_q95_endpoint_mean_zero": nan.copy(),
            "var_endpoint_mean_zero": nan.copy(),
            "pmf_forward_source_zero": nan.copy(),
            "pmf_reverse_target_zero": nan.copy(),
            "boot_stack_forward_source_zero": np.empty((0, segment_grid.size), dtype=float),
            "boot_stack_reverse_target_zero": np.empty((0, segment_grid.size), dtype=float),
            "boot_mean_forward_source_zero": nan.copy(),
            "boot_mean_reverse_target_zero": nan.copy(),
            "boot_q05_forward_source_zero": nan.copy(),
            "boot_q95_forward_source_zero": nan.copy(),
            "boot_q05_reverse_target_zero": nan.copy(),
            "boot_q95_reverse_target_zero": nan.copy(),
            "var_forward_source_zero": nan.copy(),
            "var_reverse_target_zero": nan.copy(),
            "var_min_directional": nan.copy(),
            "var_min_oneway": nan.copy(),
            "variance_alignment": "min_bidirectional_source_zero_target_zero",
            "n_boot": 0,
        }

    pmf_endpoint_mean_zero = align_pmf_to_endpoint_average_zero(pmf_ref0, grid, start_anchor, end_anchor)
    endpoint_mean_zero_stack = []
    for sample in boot_stack:
        endpoint_mean_zero = align_pmf_to_endpoint_average_zero(sample, grid, start_anchor, end_anchor)
        endpoint_mean_zero_stack.append(
            masked_interval(endpoint_mean_zero, grid, start_anchor, end_anchor)[segment_mask]
        )
    endpoint_mean_zero_stack = np.vstack(endpoint_mean_zero_stack)
    ddof = 1 if endpoint_mean_zero_stack.shape[0] > 1 else 0
    with np.errstate(invalid="ignore"):
        var_endpoint_mean_zero = np.nanvar(endpoint_mean_zero_stack, axis=0, ddof=ddof)
        boot_mean_endpoint_mean_zero = np.nanmean(endpoint_mean_zero_stack, axis=0)
        boot_q05_endpoint_mean_zero = np.nanpercentile(endpoint_mean_zero_stack, 5.0, axis=0)
        boot_q95_endpoint_mean_zero = np.nanpercentile(endpoint_mean_zero_stack, 95.0, axis=0)
    nan = np.full(segment_grid.shape, np.nan, dtype=float)
    return {
        "grid": segment_grid,
        "mask": segment_mask,
        "pmf_endpoint_mean_zero": masked_interval(
            pmf_endpoint_mean_zero,
            grid,
            start_anchor,
            end_anchor,
        )[segment_mask],
        "boot_stack_endpoint_mean_zero": endpoint_mean_zero_stack,
        "boot_mean_endpoint_mean_zero": boot_mean_endpoint_mean_zero,
        "boot_q05_endpoint_mean_zero": boot_q05_endpoint_mean_zero,
        "boot_q95_endpoint_mean_zero": boot_q95_endpoint_mean_zero,
        "var_endpoint_mean_zero": var_endpoint_mean_zero,
        "pmf_forward_source_zero": masked_interval(
            pmf_endpoint_mean_zero,
            grid,
            start_anchor,
            end_anchor,
        )[segment_mask],
        "pmf_reverse_target_zero": masked_interval(
            pmf_endpoint_mean_zero,
            grid,
            start_anchor,
            end_anchor,
        )[segment_mask],
        "boot_stack_forward_source_zero": endpoint_mean_zero_stack,
        "boot_stack_reverse_target_zero": endpoint_mean_zero_stack.copy(),
        "boot_mean_forward_source_zero": nan.copy(),
        "boot_mean_reverse_target_zero": nan.copy(),
        "boot_q05_forward_source_zero": nan.copy(),
        "boot_q95_forward_source_zero": nan.copy(),
        "boot_q05_reverse_target_zero": nan.copy(),
        "boot_q95_reverse_target_zero": nan.copy(),
        "var_forward_source_zero": var_endpoint_mean_zero.copy(),
        "var_reverse_target_zero": var_endpoint_mean_zero.copy(),
        "var_min_directional": var_endpoint_mean_zero.copy(),
        "var_min_oneway": var_endpoint_mean_zero.copy(),
        "variance_alignment": "fallback_endpoint_average_zero",
        "n_boot": int(endpoint_mean_zero_stack.shape[0]),
    }


def fragment_segmentwise_rmse_profile(record, start_anchor, end_anchor, reference_grid, reference_curve):
    grid = np.asarray(record["grid"], dtype=float)
    reference_grid = np.asarray(reference_grid, dtype=float)
    reference_curve = np.asarray(reference_curve, dtype=float)
    if len(grid) != len(reference_grid) or not np.allclose(grid, reference_grid, equal_nan=True):
        raise RuntimeError("Fragment grids must match the stitched grid.")

    pmf_ref0 = np.asarray(record["mts_pmf_ref0"], dtype=float)
    segment_mask = (
        interval_coverage_mask(grid, start_anchor, end_anchor)
        & np.isfinite(reference_curve)
        & np.isfinite(pmf_ref0)
    )
    segment_grid = grid[segment_mask]
    segment_ref = reference_curve[segment_mask]
    segment_pmf = pmf_ref0[segment_mask]
    boot_stack = np.asarray(record["mts_boot_stack"], dtype=float)
    if boot_stack.ndim == 1:
        boot_stack = boot_stack[None, :]
    segment_boot_stack = (
        boot_stack[:, segment_mask]
        if boot_stack.size and segment_grid.size
        else np.empty((0, segment_grid.size), dtype=float)
    )
    if segment_grid.size == 0:
        nan = np.full(segment_grid.shape, np.nan, dtype=float)
        return {
            "grid": segment_grid,
            "mask": segment_mask,
            "segment_ref": segment_ref,
            "segment_pmf": nan.copy(),
            "pmf_boot_low": nan.copy(),
            "pmf_boot_high": nan.copy(),
            "pmf_min_rmse": nan.copy(),
            "shift": float("nan"),
            "rmse": float("nan"),
        }

    segment_pmf, shift = align_to_reference_min_rmse(segment_pmf, segment_ref)
    if segment_boot_stack.size == 0:
        segment_boot_sigma = np.full(segment_grid.shape, np.nan, dtype=float)
    else:
        ddof = 1 if segment_boot_stack.shape[0] > 1 else 0
        with np.errstate(invalid="ignore"):
            segment_boot_sigma = np.nanstd(segment_boot_stack, axis=0, ddof=ddof)
    segment_boot_low = np.asarray(segment_pmf, dtype=float) - np.asarray(segment_boot_sigma, dtype=float)
    segment_boot_high = np.asarray(segment_pmf, dtype=float) + np.asarray(segment_boot_sigma, dtype=float)
    return {
        "grid": segment_grid,
        "mask": segment_mask,
        "segment_ref": segment_ref,
        "segment_pmf": np.asarray(segment_pmf, dtype=float),
        "pmf_boot_low": np.asarray(segment_boot_low, dtype=float),
        "pmf_boot_high": np.asarray(segment_boot_high, dtype=float),
        "pmf_min_rmse": np.asarray(segment_pmf, dtype=float),
        "shift": float(shift),
        "rmse": float(rmse(segment_pmf, segment_ref)),
    }
