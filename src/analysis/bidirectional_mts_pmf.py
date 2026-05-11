"""Bidirectional MTS PMF helpers for MiNES.

This module follows the implementation note in
`Theory/bidirectional_mts_pmf_implementation.md`.

The important correction relative to the older notebook-local implementation is:

- keep the usual multiple-time-slice (MTS) outer denominator
- replace the old forward-only single-slice numerator with the bidirectional
  BAR/EBS-weighted numerator that uses both forward and reverse trajectories

The module is intentionally small and explicit so the notebook builders can
import it directly and the code can be compared line-by-line with the Theory
note.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def trajectory_frames_to_arrays(
    trajs: Iterable[pd.DataFrame],
    n_time: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert saved NEQ trajectory frames into dense x/work arrays.

    The notebook builders store each trajectory as a small CSV with at least
    columns `x` and `work`. For the PMF code we want shape `(n_traj, n_time)`
    arrays so each time slice can be handled uniformly.
    """

    traj_list = list(trajs)
    if not traj_list:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)

    if n_time is None:
        n_time = min(len(df) for df in traj_list)
    if n_time <= 0:
        return np.empty((len(traj_list), 0), dtype=float), np.empty(
            (len(traj_list), 0), dtype=float
        )

    x = np.array(
        [df["x"].to_numpy(dtype=float)[:n_time] for df in traj_list],
        dtype=float,
    )
    work = np.array(
        [df["work"].to_numpy(dtype=float)[:n_time] for df in traj_list],
        dtype=float,
    )
    return x, work


def align_pmf_to_reference(
    pmf: np.ndarray,
    grid: np.ndarray,
    reference_x: float,
    *,
    reference_value: float = 0.0,
) -> np.ndarray:
    """Shift a PMF so the selected reference bin takes the requested value."""

    pmf = np.asarray(pmf, dtype=float).copy()
    grid = np.asarray(grid, dtype=float)
    if pmf.ndim != 1 or grid.ndim != 1 or len(pmf) != len(grid) or len(grid) == 0:
        return np.full(len(grid), np.nan, dtype=float)
    ref_idx = int(np.argmin(np.abs(grid - float(reference_x))))
    if not np.isfinite(pmf[ref_idx]):
        return np.full(len(grid), np.nan, dtype=float)
    finite = np.isfinite(pmf)
    pmf[finite] -= float(pmf[ref_idx])
    pmf[finite] += float(reference_value)
    return pmf


def align_pmf_to_endpoint_average_zero(
    pmf: np.ndarray,
    grid: np.ndarray,
    left_x: float,
    right_x: float,
) -> np.ndarray:
    """Shift a PMF so the average of the two endpoint bins is zero."""

    pmf = np.asarray(pmf, dtype=float).copy()
    grid = np.asarray(grid, dtype=float)
    if pmf.ndim != 1 or grid.ndim != 1 or len(pmf) != len(grid) or len(grid) == 0:
        return np.full(len(grid), np.nan, dtype=float)
    left_idx = int(np.argmin(np.abs(grid - float(left_x))))
    right_idx = int(np.argmin(np.abs(grid - float(right_x))))
    if not np.isfinite(pmf[left_idx]) or not np.isfinite(pmf[right_idx]):
        return np.full(len(grid), np.nan, dtype=float)
    finite = np.isfinite(pmf)
    endpoint_mean = 0.5 * float(pmf[left_idx] + pmf[right_idx])
    pmf[finite] -= endpoint_mean
    return pmf


def single_slice_from_work(work: np.ndarray) -> np.ndarray:
    """Forward Jarzynski estimate for each time slice.

    For an array of cumulative works `work[:, t]`, this returns

        -log <exp(-W_t)>

    slice by slice. The Theory note treats the intermediate reduced free
    energies `f_t` as an input. In practice the notebooks still need a concrete
    estimate of `f_t`, so we retain the existing bidirectional slice-free-energy
    approximation and only change the PMF numerator.
    """

    work = np.asarray(work, dtype=float)
    if work.ndim != 2:
        raise ValueError("work must have shape (n_traj, n_time)")
    if work.shape[0] == 0:
        return np.empty(work.shape[1], dtype=float)
    return -(logsumexp(-work, axis=0) - math.log(float(work.shape[0])))


def estimate_intermediate_reduced_free_energies(
    forward_work: np.ndarray,
    reverse_prefix_work_aligned: np.ndarray,
    delta_f_0T: float,
    *,
    n_boot: int = 64,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate `f_t` for the intermediate biased states.

    Inputs:
    - `forward_work[:, t]` is the forward cumulative work from state A to the
      matched intermediate state at time slice `t`
    - `reverse_prefix_work_aligned[:, t]` is the reverse cumulative work from
      state B to that same matched intermediate state. In practice this is the
      reverse cumulative work array reversed in time, so column `t` corresponds
      to the forward matched slice.

    The Theory note leaves the source of `f_t` open. The current MiNES notebook
    continues to estimate these intermediate reduced free energies by combining:
    - the forward Jarzynski slice estimate
    - the reverse slice estimate shifted by the endpoint `DeltaF`

    with inverse-variance weighting from simple bootstrap resampling.
    """

    forward_work = np.asarray(forward_work, dtype=float)
    reverse_prefix_work_aligned = np.asarray(reverse_prefix_work_aligned, dtype=float)
    if forward_work.ndim != 2 or reverse_prefix_work_aligned.ndim != 2:
        raise ValueError("forward_work and reverse_prefix_work_aligned must be 2D")
    if forward_work.shape[1] != reverse_prefix_work_aligned.shape[1]:
        raise ValueError("forward and reverse slice-work arrays must share n_time")

    n_fwd, n_time = forward_work.shape
    n_rev, _ = reverse_prefix_work_aligned.shape

    f_fwd = single_slice_from_work(forward_work)
    f_rev = float(delta_f_0T) + single_slice_from_work(reverse_prefix_work_aligned)

    if n_fwd == 0 or n_rev == 0:
        nan = np.full(n_time, np.nan, dtype=float)
        return f_fwd, f_rev, f_fwd, nan, nan, nan

    rng = np.random.default_rng(rng_seed)
    f_fwd_boot = np.empty((n_boot, n_time), dtype=float)
    f_rev_boot = np.empty((n_boot, n_time), dtype=float)
    for boot_idx in range(n_boot):
        idx_fwd = rng.integers(0, n_fwd, size=n_fwd)
        idx_rev = rng.integers(0, n_rev, size=n_rev)
        f_fwd_boot[boot_idx] = single_slice_from_work(forward_work[idx_fwd])
        f_rev_boot[boot_idx] = float(delta_f_0T) + single_slice_from_work(
            reverse_prefix_work_aligned[idx_rev]
        )

    sig_fwd = f_fwd_boot.std(axis=0, ddof=1)
    sig_rev = f_rev_boot.std(axis=0, ddof=1)
    eps = 1.0e-16
    weight_fwd = 1.0 / (sig_fwd**2 + eps)
    weight_rev = 1.0 / (sig_rev**2 + eps)
    f_bi = (f_fwd * weight_fwd + f_rev * weight_rev) / (weight_fwd + weight_rev)

    f_bi_boot = (
        f_fwd_boot * weight_fwd[None, :] + f_rev_boot * weight_rev[None, :]
    ) / (weight_fwd[None, :] + weight_rev[None, :])
    sig_bi = f_bi_boot.std(axis=0, ddof=1)
    return f_fwd, f_rev, f_bi, sig_fwd, sig_rev, sig_bi


def parse_bar_result(result: object) -> tuple[float, float]:
    """Normalize PyMBAR BAR outputs across dict/tuple/scalar return styles."""

    if isinstance(result, dict):
        return float(result["Delta_f"]), float(result.get("dDelta_f", float("nan")))
    if isinstance(result, tuple):
        return float(result[0]), float(result[1]) if len(result) > 1 else float("nan")
    return float(result), float("nan")


def solve_segment_cft_delta_f_once(
    work_forward: np.ndarray,
    work_reverse: np.ndarray,
    *,
    kT: float = 1.0,
) -> dict[str, float | bool | str | None]:
    """Solve the endpoint bidirectional free-energy difference once.

    The current implementation uses the same PyMBAR BAR solve that the full
    bootstrap path already relies on. This helper only standardizes the
    success/failure contract for callers that want to reuse a fixed `DeltaF`
    through multiple bootstrap reconstructions.
    """

    del kT  # kept for a stable caller-facing interface

    from pymbar import other_estimators

    work_forward = np.asarray(work_forward, dtype=float)
    work_reverse = np.asarray(work_reverse, dtype=float)
    if work_forward.ndim != 2 or work_reverse.ndim != 2:
        return {
            "cft_solved": False,
            "delta_f": None,
            "delta_f_unc": None,
            "method": "BAR",
            "reason": "work_arrays_must_be_2d",
        }
    if work_forward.shape[0] <= 0 or work_reverse.shape[0] <= 0:
        return {
            "cft_solved": False,
            "delta_f": None,
            "delta_f_unc": None,
            "method": "BAR",
            "reason": "missing_forward_or_reverse_trajectories",
        }
    n_time = min(work_forward.shape[1], work_reverse.shape[1])
    if n_time <= 0:
        return {
            "cft_solved": False,
            "delta_f": None,
            "delta_f_unc": None,
            "method": "BAR",
            "reason": "no_time_slices",
        }
    try:
        delta_f, delta_f_unc = parse_bar_result(
            other_estimators.bar(
                work_forward[:, n_time - 1],
                work_reverse[:, n_time - 1],
                compute_uncertainty=True,
            )
        )
    except Exception as exc:
        return {
            "cft_solved": False,
            "delta_f": None,
            "delta_f_unc": None,
            "method": "BAR",
            "reason": f"bar_failed: {exc}",
        }
    if not math.isfinite(float(delta_f)):
        return {
            "cft_solved": False,
            "delta_f": None,
            "delta_f_unc": None,
            "method": "BAR",
            "reason": "bar_returned_nonfinite_delta_f",
        }
    if not math.isfinite(float(delta_f_unc)):
        delta_f_unc = float("nan")
    return {
        "cft_solved": True,
        "delta_f": float(delta_f),
        "delta_f_unc": float(delta_f_unc),
        "method": "BAR",
        "reason": "ok",
    }


def build_bidirectional_mts_pmf(
    x_forward: np.ndarray,
    work_forward: np.ndarray,
    x_reverse: np.ndarray,
    work_reverse: np.ndarray,
    centers: np.ndarray,
    k_values: np.ndarray,
    grid: np.ndarray,
    fk: np.ndarray,
    delta_f_0T: float,
    *,
    kT: float = 1.0,
) -> np.ndarray:
    """Build the bidirectional MTS PMF on a uniform reaction-coordinate grid.

    This function mirrors the Theory note step-by-step:

    1. For each time slice `t`, build the bidirectional numerator histogram
       `N_t(x_m)` using:
       - forward contributions at the same slice `t`
       - reverse contributions at the matched reverse slice `tau - t`
       - BAR/EBS-style denominators with the endpoint `DeltaF`
    2. Combine those `N_t(x_m)` with the usual Hummer-Szabo MTS denominator
       `sum_t exp(f_t - beta V_t(x_m))`
    3. Normalize the resulting density and convert it to a PMF

    All accumulation is done in log space wherever it matters numerically.
    """

    x_forward = np.asarray(x_forward, dtype=float)
    work_forward = np.asarray(work_forward, dtype=float)
    x_reverse = np.asarray(x_reverse, dtype=float)
    work_reverse = np.asarray(work_reverse, dtype=float)
    centers = np.asarray(centers, dtype=float)
    k_values = np.asarray(k_values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    fk = np.asarray(fk, dtype=float)

    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("grid must be a 1D uniform grid with at least two bins")

    n_time = min(
        x_forward.shape[1] if x_forward.ndim == 2 and x_forward.size else 0,
        work_forward.shape[1] if work_forward.ndim == 2 and work_forward.size else 0,
        x_reverse.shape[1] if x_reverse.ndim == 2 and x_reverse.size else 0,
        work_reverse.shape[1] if work_reverse.ndim == 2 and work_reverse.size else 0,
        len(centers),
        len(k_values),
        len(fk),
    )
    if n_time <= 0:
        return np.full(len(grid), np.nan, dtype=float)

    x_forward = x_forward[:, :n_time]
    work_forward = work_forward[:, :n_time]
    x_reverse = x_reverse[:, :n_time]
    work_reverse = work_reverse[:, :n_time]
    centers = centers[:n_time]
    k_values = k_values[:n_time]
    fk = fk[:n_time]

    n_fwd = x_forward.shape[0]
    n_rev = x_reverse.shape[0]
    if n_fwd == 0 or n_rev == 0:
        return np.full(len(grid), np.nan, dtype=float)

    beta = 1.0 / float(kT)
    dx = abs(float(grid[1] - grid[0]))
    x_left = float(grid[0])
    m_bins = len(grid)

    wf_tot = work_forward[:, n_time - 1]
    wr_tot = work_reverse[:, n_time - 1]
    log_nf = math.log(float(n_fwd))
    log_nr = math.log(float(n_rev))
    log_dx = math.log(dx)

    # These denominators are constant over time for each trajectory because the
    # Theory-note formulas use the total endpoint work in the BAR/EBS factor.
    log_denom_forward = np.logaddexp(
        log_nf,
        log_nr - beta * (wf_tot - float(delta_f_0T)),
    )
    log_denom_reverse = np.logaddexp(
        log_nf,
        log_nr + beta * (wr_tot + float(delta_f_0T)),
    )

    # Outer MTS numerator: log sum_t exp(f_t) * N_t(x_m)
    log_numerator = np.full(m_bins, -np.inf, dtype=float)

    for time_idx in range(n_time):
        log_nt = np.full(m_bins, -np.inf, dtype=float)

        # Forward contribution uses the same time slice t.
        x_t_forward = x_forward[:, time_idx]
        w_t_forward = work_forward[:, time_idx]
        idx_forward = np.rint((x_t_forward - x_left) / dx).astype(int)
        valid_forward = (idx_forward >= 0) & (idx_forward < m_bins)
        valid_forward &= (
            np.abs(grid[np.clip(idx_forward, 0, m_bins - 1)] - x_t_forward)
            <= 0.5 * dx + 1.0e-9
        )
        forward_log_contrib = -beta * w_t_forward - log_denom_forward - log_dx
        for bin_idx, log_val in zip(idx_forward[valid_forward], forward_log_contrib[valid_forward]):
            log_nt[bin_idx] = np.logaddexp(log_nt[bin_idx], log_val)

        # Reverse contribution uses the matched reverse slice tau - t.
        reverse_idx = n_time - 1 - time_idx
        x_t_reverse = x_reverse[:, reverse_idx]
        tail_work = wr_tot - work_reverse[:, reverse_idx]
        idx_reverse = np.rint((x_t_reverse - x_left) / dx).astype(int)
        valid_reverse = (idx_reverse >= 0) & (idx_reverse < m_bins)
        valid_reverse &= (
            np.abs(grid[np.clip(idx_reverse, 0, m_bins - 1)] - x_t_reverse)
            <= 0.5 * dx + 1.0e-9
        )
        reverse_log_contrib = beta * tail_work - log_denom_reverse - log_dx
        for bin_idx, log_val in zip(idx_reverse[valid_reverse], reverse_log_contrib[valid_reverse]):
            log_nt[bin_idx] = np.logaddexp(log_nt[bin_idx], log_val)

        log_numerator = np.logaddexp(log_numerator, fk[time_idx] + log_nt)

    # Standard MTS denominator: sum_t exp(f_t - beta * V_t(x))
    harmonic_terms = np.empty((n_time, m_bins), dtype=float)
    for time_idx in range(n_time):
        harmonic_terms[time_idx] = fk[time_idx] - beta * 0.5 * k_values[time_idx] * (
            grid - centers[time_idx]
        ) ** 2
    log_denominator = logsumexp(harmonic_terms, axis=0)

    log_rho = log_numerator - log_denominator
    finite = np.isfinite(log_rho)
    pmf = np.full(m_bins, np.nan, dtype=float)
    if np.any(finite):
        log_norm = float(logsumexp(log_rho[finite])) + log_dx
        log_rho[finite] -= log_norm
        pmf[finite] = -float(kT) * log_rho[finite]
        pmf[finite] -= np.nanmin(pmf[finite])
    return pmf


def bootstrap_bidirectional_mts_pmf(
    x_forward: np.ndarray,
    work_forward: np.ndarray,
    x_reverse: np.ndarray,
    work_reverse: np.ndarray,
    centers: np.ndarray,
    k_values: np.ndarray,
    grid: np.ndarray,
    *,
    reference_x: float,
    kT: float = 1.0,
    n_boot: int = 32,
    fk_boot: int = 8,
    rng_seed: int = 0,
    fixed_delta_f: float | None = None,
    recompute_delta_f_per_bootstrap: bool = True,
) -> dict[str, np.ndarray | float | int]:
    """Bootstrap the bidirectional MTS PMF relative to a reference anchor.

    Each bootstrap replicate resamples whole forward and reverse trajectories,
    re-estimates the intermediate reduced free energies `f_t`, reconstructs
    the bidirectional MTS PMF, and finally shifts it so the reference bin at
    `reference_x` is zero. When `fixed_delta_f` is finite and
    `recompute_delta_f_per_bootstrap` is false, the same endpoint free-energy
    difference is reused across every bootstrap replicate instead of solving
    BAR again each time.
    """

    from pymbar import other_estimators

    x_forward = np.asarray(x_forward, dtype=float)
    work_forward = np.asarray(work_forward, dtype=float)
    x_reverse = np.asarray(x_reverse, dtype=float)
    work_reverse = np.asarray(work_reverse, dtype=float)
    centers = np.asarray(centers, dtype=float)
    k_values = np.asarray(k_values, dtype=float)
    grid = np.asarray(grid, dtype=float)

    n_time = min(
        x_forward.shape[1] if x_forward.ndim == 2 and x_forward.size else 0,
        work_forward.shape[1] if work_forward.ndim == 2 and work_forward.size else 0,
        x_reverse.shape[1] if x_reverse.ndim == 2 and x_reverse.size else 0,
        work_reverse.shape[1] if work_reverse.ndim == 2 and work_reverse.size else 0,
        len(centers),
        len(k_values),
    )
    if n_time <= 0:
        nan = np.full(len(grid), np.nan, dtype=float)
        return {
            "delta_f": float("nan"),
            "delta_f_unc": float("nan"),
            "pmf": nan.copy(),
            "pmf_ref0": nan.copy(),
            "boot_pmf_stack": np.empty((0, len(grid)), dtype=float),
            "std_ref0": nan.copy(),
            "var_ref0": nan.copy(),
            "q05_ref0": nan.copy(),
            "q95_ref0": nan.copy(),
            "n_boot_used": 0,
            "bootstrap_recomputed_cft": int(bool(recompute_delta_f_per_bootstrap)),
        }

    x_forward = x_forward[:, :n_time]
    work_forward = work_forward[:, :n_time]
    x_reverse = x_reverse[:, :n_time]
    work_reverse = work_reverse[:, :n_time]
    centers = centers[:n_time]
    k_values = k_values[:n_time]

    n_fwd = x_forward.shape[0]
    n_rev = x_reverse.shape[0]
    if n_fwd == 0 or n_rev == 0:
        nan = np.full(len(grid), np.nan, dtype=float)
        return {
            "delta_f": float("nan"),
            "delta_f_unc": float("nan"),
            "pmf": nan.copy(),
            "pmf_ref0": nan.copy(),
            "boot_pmf_stack": np.empty((0, len(grid)), dtype=float),
            "std_ref0": nan.copy(),
            "var_ref0": nan.copy(),
            "q05_ref0": nan.copy(),
            "q95_ref0": nan.copy(),
            "n_boot_used": 0,
            "bootstrap_recomputed_cft": int(bool(recompute_delta_f_per_bootstrap)),
        }

    if fixed_delta_f is not None and math.isfinite(float(fixed_delta_f)):
        solved = solve_segment_cft_delta_f_once(
            work_forward[:, :n_time],
            work_reverse[:, :n_time],
            kT=kT,
        )
        delta_f = float(fixed_delta_f)
        delta_f_unc = (
            float(solved["delta_f_unc"])
            if solved.get("delta_f_unc") is not None and math.isfinite(float(solved["delta_f_unc"]))
            else float("nan")
        )
    else:
        delta_f, delta_f_unc = parse_bar_result(
            other_estimators.bar(
                work_forward[:, n_time - 1],
                work_reverse[:, n_time - 1],
                compute_uncertainty=True,
            )
        )
    reverse_prefix = work_reverse[:, ::-1][:, :n_time]
    _, _, fk, _, _, _ = estimate_intermediate_reduced_free_energies(
        work_forward[:, :n_time],
        reverse_prefix,
        delta_f,
        n_boot=max(int(fk_boot), 4),
        rng_seed=int(rng_seed),
    )
    pmf = build_bidirectional_mts_pmf(
        x_forward,
        work_forward,
        x_reverse,
        work_reverse,
        centers,
        k_values,
        grid,
        fk[:n_time],
        delta_f,
        kT=kT,
    )
    pmf_ref0 = align_pmf_to_reference(pmf, grid, reference_x, reference_value=0.0)

    rng = np.random.default_rng(rng_seed)
    boot_pmfs: list[np.ndarray] = []
    for boot_idx in range(int(n_boot)):
        idx_f = rng.integers(0, n_fwd, size=n_fwd)
        idx_r = rng.integers(0, n_rev, size=n_rev)
        x_f_boot = x_forward[idx_f]
        w_f_boot = work_forward[idx_f]
        x_r_boot = x_reverse[idx_r]
        w_r_boot = work_reverse[idx_r]
        try:
            if fixed_delta_f is not None and math.isfinite(float(fixed_delta_f)) and not bool(
                recompute_delta_f_per_bootstrap
            ):
                delta_f_boot = float(fixed_delta_f)
            else:
                delta_f_boot, _ = parse_bar_result(
                    other_estimators.bar(
                        w_f_boot[:, n_time - 1],
                        w_r_boot[:, n_time - 1],
                        compute_uncertainty=True,
                    )
                )
            reverse_prefix_boot = w_r_boot[:, ::-1][:, :n_time]
            _, _, fk_bootstrap, _, _, _ = estimate_intermediate_reduced_free_energies(
                w_f_boot[:, :n_time],
                reverse_prefix_boot,
                delta_f_boot,
                n_boot=max(int(fk_boot), 4),
                rng_seed=int(rng_seed + 1000 + boot_idx),
            )
            pmf_boot = build_bidirectional_mts_pmf(
                x_f_boot,
                w_f_boot,
                x_r_boot,
                w_r_boot,
                centers,
                k_values,
                grid,
                fk_bootstrap[:n_time],
                delta_f_boot,
                kT=kT,
            )
            pmf_boot_ref0 = align_pmf_to_reference(
                pmf_boot, grid, reference_x, reference_value=0.0
            )
            if np.any(np.isfinite(pmf_boot_ref0)):
                boot_pmfs.append(pmf_boot_ref0)
        except Exception:
            continue

    if not boot_pmfs:
        nan = np.full(len(grid), np.nan, dtype=float)
        return {
            "delta_f": float(delta_f),
            "delta_f_unc": float(delta_f_unc),
            "pmf": pmf.copy(),
            "pmf_ref0": pmf_ref0.copy(),
            "boot_pmf_stack": np.empty((0, len(grid)), dtype=float),
            "std_ref0": nan.copy(),
            "var_ref0": nan.copy(),
            "q05_ref0": nan.copy(),
            "q95_ref0": nan.copy(),
            "n_boot_used": 0,
            "bootstrap_recomputed_cft": int(bool(recompute_delta_f_per_bootstrap)),
        }

    stack = np.vstack(boot_pmfs)
    with np.errstate(invalid="ignore"):
        boot_mean = np.nanmean(stack, axis=0)
        boot_var = np.nanvar(stack, axis=0, ddof=1 if stack.shape[0] > 1 else 0)
        boot_std = np.sqrt(boot_var)
        boot_q05 = np.nanpercentile(stack, 5.0, axis=0)
        boot_q95 = np.nanpercentile(stack, 95.0, axis=0)

    return {
        "delta_f": float(delta_f),
        "delta_f_unc": float(delta_f_unc),
        "pmf": pmf.copy(),
        "pmf_ref0": pmf_ref0.copy(),
        "boot_pmf_stack": np.asarray(stack, dtype=float),
        "boot_mean_ref0": np.asarray(boot_mean, dtype=float),
        "std_ref0": np.asarray(boot_std, dtype=float),
        "var_ref0": np.asarray(boot_var, dtype=float),
        "q05_ref0": np.asarray(boot_q05, dtype=float),
        "q95_ref0": np.asarray(boot_q95, dtype=float),
        "n_boot_used": int(stack.shape[0]),
        "bootstrap_recomputed_cft": int(bool(recompute_delta_f_per_bootstrap)),
    }
