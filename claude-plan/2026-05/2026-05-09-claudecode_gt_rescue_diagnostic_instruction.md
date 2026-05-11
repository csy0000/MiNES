# ClaudeCode Task: Add Minimal GT Rescue Harmonic Diagnostics to the MiNES Notebook

## Goal

Add a focused diagnostic visualization for each rescue window generated using GT mode. The diagnostic should show only the three harmonic background approximations that are needed to debug the rescue placement:

1. The harmonic approximation inferred from the left anchor window, corresponding to `s = 0`.
2. The harmonic approximation inferred from the right anchor window, corresponding to `s = 1`.
3. The harmonic approximation at the GT-selected target interpolation point, using the selected `s` whose `m_target` / `ms` is closest to `target_bin_x`.

This should make it easy to see whether the GT harmonic interpolation is producing a reasonable local background at the rescue target, or whether it is underestimating the force/curvature and therefore creating a rescue umbrella that samples away from the target bin.

## Files to modify

Primary:

- `notebooks/mines_variance_fusion_visualization.ipynb`

Secondary, only if required because the needed fields are not already written:

- `scripts/mines_variance_fusion.py`

The notebook is the main target. Avoid changing the simulation logic unless the diagnostic metadata is missing from disk outputs.

---

## Existing useful outputs

The current implementation already writes GT protocol diagnostics for each NEQ segment under paths like:

```text
segments/<SEG_NAME>/protocols/forward_protocol_diagnostics.csv
segments/<SEG_NAME>/protocols/reverse_protocol_diagnostics.csv
segments/<SEG_NAME>/protocols/protocol_summary.json
```

The GT diagnostics rows should contain fields such as:

```text
s
x_raw
k_raw
x
k
m_target
sigma_target
K_target
x0_s
k0_s
x0_L
k0_L
x0_R
k0_R
mL
sigmaL
mR
sigmaR
KL
KR
```

The rescue summary should contain fields such as:

```text
window
target_bin_x
rescue_center_x
left_window
right_window
segment or gap information if available
```

Use the exact available column names from the current output files. If names differ, adapt robustly.

---

## Required notebook section

Add a new section titled:

```markdown
## GT rescue harmonic diagnostics
```

This section should:

1. Load `rescue_summary.csv`.
2. Load `windows.csv`.
3. For each rescue window, identify the left and right anchor windows used by GT.
4. Load the corresponding `forward_protocol_diagnostics.csv` or `reverse_protocol_diagnostics.csv` for the segment that generated the GT bridge relevant to that rescue.
5. Select the diagnostic row with `s` such that `m_target` or `ms` is closest to `target_bin_x`.
6. Plot three harmonic curves:
   - left background harmonic, `s = 0`
   - right background harmonic, `s = 1`
   - selected target harmonic, `s = s_target`
7. Overlay the read/global PMF if available.
8. Mark the target bin and rescue center.

---

## Harmonic curves to plot

Use the inferred background harmonic:

```python
F_harm(x; k0, x0) = 0.5 * k0 * (x - x0)**2
```

For the left curve:

```python
k0 = k0_L
x0 = x0_L
s = 0
```

For the right curve:

```python
k0 = k0_R
x0 = x0_R
s = 1
```

For the selected target curve:

```python
k0 = k0_s
x0 = x0_s
s = selected_s
```

Because these harmonic curves can have arbitrary vertical offsets relative to the global PMF, align them before plotting.

Preferred alignment:

- If the global PMF is finite at `target_bin_x`, shift all three harmonic curves so they equal the global PMF at `target_bin_x`.
- Otherwise, shift each harmonic curve to have minimum zero over the local plotting range.

Use a local plotting range around the rescue gap, for example:

```python
xlo = min(left_mean_x, right_mean_x, target_bin_x, rescue_center_x) - 1.5
xhi = max(left_mean_x, right_mean_x, target_bin_x, rescue_center_x) + 1.5
```

Clip this range to the analysis bounds if available.

---

## Plot content

For each rescue window, generate one figure with:

### Main panel

Plot:

```text
global/read PMF                       black or blue solid line
left harmonic background, s = 0       dashed line
right harmonic background, s = 1      dashed line
target harmonic, s = selected_s       thicker solid or dash-dot line
target_bin_x                          red vertical dotted line
rescue_center_x                       orange vertical dash-dot line
sampled rescue mean_x                 green vertical line if available
rescue tail q05-q95                   translucent band if available
```

Title format:

```text
<window>: GT harmonic diagnostics, target x=<target_bin_x>, selected s=<s_target>
```

### Optional small table printed below or beside the plot

Include:

```text
window
target_bin_x
rescue_center_x
mean_x
k_rescue
left_window
right_window
selected_s
m_target or ms
abs(m_target - target_bin_x)
k0_L, x0_L
k0_R, x0_R
k0_s, x0_s
k_raw, k_clipped
x_raw, x_clipped
```

This table is important because the main debugging question is whether `selected_s` is close to 0/1 and whether `m_target` actually approaches the target bin.

---

## Robust helper functions to add to the notebook

Add helper functions similar to these. Adapt names as needed.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_optional(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_optional(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def harmonic_curve(x, k0, x0):
    return 0.5 * float(k0) * (x - float(x0))**2


def align_curve_to_pmf_at_x(x_grid, y_curve, pmf_df, x_ref):
    if pmf_df is None or pmf_df.empty or "global_pmf" not in pmf_df.columns:
        finite = np.isfinite(y_curve)
        if finite.any():
            y_curve = y_curve.copy()
            y_curve[finite] -= np.nanmin(y_curve[finite])
        return y_curve

    pmf_x = pmf_df["x"].to_numpy(float)
    pmf_y = pmf_df["global_pmf"].to_numpy(float)
    idx = int(np.nanargmin(np.abs(pmf_x - float(x_ref))))
    if not np.isfinite(pmf_y[idx]):
        finite = np.isfinite(y_curve)
        if finite.any():
            y_curve = y_curve.copy()
            y_curve[finite] -= np.nanmin(y_curve[finite])
        return y_curve

    curve_idx = int(np.nanargmin(np.abs(x_grid - float(x_ref))))
    if np.isfinite(y_curve[curve_idx]):
        return y_curve - y_curve[curve_idx] + pmf_y[idx]
    return y_curve


def select_gt_target_row(proto_df, target_bin_x):
    if proto_df.empty:
        return None
    if "m_target" in proto_df.columns:
        key = "m_target"
    elif "ms" in proto_df.columns:
        key = "ms"
    else:
        return None
    values = pd.to_numeric(proto_df[key], errors="coerce")
    valid = np.isfinite(values.to_numpy(float))
    if not valid.any():
        return None
    idx = np.nanargmin(np.abs(values.to_numpy(float) - float(target_bin_x)))
    return proto_df.iloc[int(idx)]
```

---

## Segment/protocol matching logic

Implement a robust matching strategy.

Preferred order:

1. If `rescue_summary.csv` has an explicit segment/protocol file column, use it directly.
2. Else, if it has `left_window` and `right_window`, look in `segments.csv` for a segment whose `boundary_left` and `boundary_right` match these windows.
3. Else, infer the nearest segment by `target_bin_x`, using segment boundary means from `windows.csv`.

For each matched segment:

- Prefer `forward_protocol_diagnostics.csv` if it goes from left boundary to right boundary.
- Use `reverse_protocol_diagnostics.csv` only if the rescue metadata indicates the reverse direction was used.
- If uncertain, choose the diagnostics file whose `m_target` range contains or comes closest to `target_bin_x`.

Implement this defensively. If a rescue window cannot be matched to a protocol file, print a clear warning and continue to the next rescue window.

---

## Important implementation detail: do not overcomplicate the plot

Do **not** plot every GT interpolation state.

For each rescue window, plot only:

```text
s = 0 left harmonic
s = 1 right harmonic
s = selected target harmonic
```

Do **not** plot all protocol steps. Do **not** add a dense spaghetti plot.

---

## If metadata is missing from the Python script

If the notebook cannot find the needed GT fields, update `scripts/mines_variance_fusion.py` to write them.

At the point where GT rescue parameters are chosen, write a file such as:

```text
rescue_round_<n>/gt_rescue_diagnostics.csv
```

or add columns to `rescue_summary.csv`:

```text
left_window
right_window
protocol_file
selected_s
m_target
sigma_target
K_target
x0_L
k0_L
x0_R
k0_R
x0_s
k0_s
k_raw
k
k_clipped
x_raw
x
x_clipped
```

The values should be copied from the same GT diagnostic row selected by minimum distance between `m_target`/`ms` and `target_bin_x`.

Acceptance condition for metadata:

For every GT rescue window, the notebook can identify the selected target row and plot all three harmonic curves without manually opening files.

---

## Acceptance criteria

1. The notebook has a new section `GT rescue harmonic diagnostics`.
2. For each rescue window, the notebook plots exactly three harmonic approximations:
   - left, `s = 0`
   - right, `s = 1`
   - target, `s = selected_s`
3. The plot overlays the global/read PMF if available.
4. The plot marks `target_bin_x` and `rescue_center_x`.
5. The notebook prints a compact diagnostic table for each rescue window.
6. The table includes `selected_s`, `m_target` or `ms`, `abs(m_target - target_bin_x)`, `k0_L`, `x0_L`, `k0_R`, `x0_R`, `k0_s`, `x0_s`, `k_raw`, `k`, `x_raw`, and `x` whenever available.
7. The notebook is robust to missing files and prints warnings rather than crashing.
8. Do not change the rescue placement logic in this task. This task is visualization/diagnostics only, except for adding missing metadata outputs.
