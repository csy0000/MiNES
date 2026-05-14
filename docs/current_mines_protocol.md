# Current MiNES Protocol

*Last updated: 2026-05-13. This file is the authoritative compact reference for the current protocol. See `README.md` for full documentation.*

---

## Exploration

### Global endpoint-anchored width profile

Computed once from first-generation endpoints L0 and R0; never updated generation by generation:

```
s(m)        = (m - m_L0) / (m_R0 - m_L0)
sigma_GT(m) = (1 - s(m)) * sigma_L0 + s(m) * sigma_R0
```

### Basin-like step (KL-GT target)

Solve for ordinary target mean `m_KL` from the Gaussian KL condition:

```
KL[ N(m_i, sigma_i^2) || N(m_KL, sigma_GT(m_KL)^2) ] = KL_target
```

`KL_target` is the configurable exploration spacing parameter.

### Transition segment detection

Fit a local variance-weighted quadratic background `F0(x) = 0.5*k0*(x-x0)^2 + C`.

Classify as **transition segment** if:

```
k0 < 0   and   min(m_i, m_KL) < x0 < max(m_i, m_KL)
```

### Transition-segment target rule (barrier reflection)

Ignore KL target; use reflected target mean:

```
m_next    = 2 * x0 - m_i
sigma_next = sigma_GT(m_next)
```

Transition segments may remain EQ-disconnected after exploration and are repaired by bidirectional NES/MTS in refinement.

### Bias parameter construction

Local harmonic inversion:

```
k_raw = 1 / (beta_eff * sigma_next^2) - k0
x_raw = ((k0 + k_raw) * m_next - k0 * x0) / k_raw
```

If `k_raw` out of bounds, clip and recompute center to preserve target mean:

```
k_child = clip(k_raw, k_min, k_max)
x_child = ((k0 + k_child) * m_next - k0 * x0) / k_child
```

Priority: (1) preserve `m_next`, (2) keep `k_child` in bounds, (3) match `sigma_next` if possible.

---

## EQ Connectivity

- Sort EQ windows by sampled mean.
- Neighboring windows are connected if BAR/MBAR pairwise overlap `O_pair >= 0.3`.
- JSD is diagnostic only; not used for connectivity decisions.
- Output: `neighbor_eq_overlap.csv`.

---

## Refinement (disconnected pairs)

**No NES truncation.**

```
If bidirectional NES already exists for the disconnected pair:
    append final perturbation work to both directions
    solve MTS
Else:
    run new bidirectional NES between the two boundary windows
    append final perturbation work
    solve MTS
```

If MTS cannot be solved, fall back to HS provisional patch or seed additional EQ refinement windows.

---

## Final Estimator

Once all EQ windows form one MBAR-connected component (`len(clusters) == 1`):

- Use **connected-EQ MBAR only** as the final PMF estimator.
- NEQ/MTS patches become diagnostics; they are **not** included in the final PMF fit.
- Final phase extends all windows by `--eq-extension-steps` per round until `max(sqrt(variance)) < --target-mbar-ddf` or budget exhaustion.
- Metadata field: `patch_selection_rule = "connected_EQ_MBAR_only"`.

---

## Key CLI Parameters

| Option | Role |
|---|---|
| `--kl-target` | KL-divergence target for basin-like step spacing |
| `--k-min`, `--k-max` | Spring constant bounds (default k_max = 100) |
| `--eq-overlap-threshold` | BAR/MBAR connectivity threshold (0.3) |
| `--pmf-method` | Provisional PMF estimator in disconnected state only |
| `--final-refinement-mode eq-extend` | Activate final EQ-extension phase |
| `--target-mbar-ddf` | Stopping criterion for final EQ extension (default 1e-3) |
| `--eq-extension-steps` | EQ steps per window per extension round |

---

## What is NOT in the current protocol

- JSD as EQ connectivity criterion (removed; diagnostic only)
- NES truncation in refinement (removed)
- Force-matching child placement (replaced by KL-GT)
- Generation-local sigma interpolation (replaced by global endpoint-anchored sigma_GT)
- s-dependent harmonic interpolation for child/rescue windows (removed)
