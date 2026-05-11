# Task: Notebook-only visualization update for MiNES

## Scope

Modify only the Jupyter notebook:

```text
notebooks/mines_variance_fusion_visualization.ipynb
```

Do **not** modify the Python MiNES scheduler or workflow code for now. In particular, do **not** change:

```text
scripts/mines_variance_fusion.py
```

Do **not** change rescue scheduling, NEQ segment scheduling, GT rescue logic, `k0/x0` inference, `k_m/x_m` design, or any budget/scheduler behavior in the Python workflow.

The current goal is only to improve notebook visualization and interactive inspection. I will separately study how `k0` and `x0` affect the design of `k_m` and `x_m` before deciding the next algorithmic change.

---

## Required notebook changes

### 1. Do not call `plt.close()`

Remove or disable all calls to:

```python
plt.close()
```

from the notebook.

Reason: I want figures to remain visible in the Jupyter notebook after each cell executes. The notebook should display plots inline and not silently close them.

Acceptable replacement:

```python
plt.show()
```

or simply leave the figure as the last object in the cell.

Do not add `plt.close(fig)` or any equivalent close operation.

---

### 2. Increase plot font sizes

Increase title, axis label, tick label, and legend font sizes by at least 2× relative to the current notebook.

Add a central plotting style cell near the top of the notebook, for example:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 24,
})
```

If individual plots override font sizes manually, update those values consistently.

---

### 3. Add grid lines to every plot

Every axis in every plot should call:

```python
ax.grid(True, alpha=0.3)
```

or an equivalent grid command.

For plots with multiple axes, apply the grid to every axis.

---

### 4. Use finer x-axis ticks

For all 1D coordinate plots where the x-axis is the reaction coordinate `x`, set x ticks using:

```python
ax.set_xticks(np.arange(x_min, x_max + 0.1, 1.0))
```

Use the current analysis range or plotted x range for `x_min` and `x_max`. If available, prefer:

```python
x_min = np.nanmin(global_pmf_df["x"])
x_max = np.nanmax(global_pmf_df["x"])
```

or the notebook's existing `analysis_xmin/analysis_xmax` values.

Make sure `numpy` is imported as:

```python
import numpy as np
```

---

### 5. Improve the window parameter table

In the table where `x_m` and `k_m` are shown, also display the following columns if available from existing output files:

```text
x_target
x0
k0
```

Also include useful nearby diagnostic columns if they already exist, for example:

```text
center_x
k
mean_x
std_x
x_most
generation
side
rescue_center_x
rescue_k
gt_x0_L
gt_k0_L
gt_x0_R
gt_k0_R
gt_s_eff
gt_k_raw
gt_x_raw
```

Important: do not change the scheduler to create new columns. The notebook should only read and display columns that already exist. If a column is missing, skip it gracefully.

Use a helper like:

```python
def select_existing_columns(df, columns):
    return [c for c in columns if c in df.columns]
```

Then display:

```python
display(df[select_existing_columns(df, desired_columns)])
```

---

## Explicit non-goals

Do **not** implement NEQ-PMF-informed rescue yet.

Do **not** change the Python MiNES scheduler.

Do **not** change how rescue windows are selected.

Do **not** change how `k0`, `x0`, `k_m`, or `x_m` are computed.

Do **not** change budget behavior.

Do **not** change which NEQ segments are executed.

Do **not** change PMF reconstruction or patch fusion.

This is a notebook-only visualization and display update.

---

## Acceptance criteria

The task is complete when:

1. The notebook contains no active `plt.close()` calls.
2. All plots remain visible after cell execution.
3. Plot fonts are at least 2× larger than before.
4. Every plot has grid lines.
5. Coordinate plots use x ticks with spacing 1.0 via `np.arange(x_min, x_max + 0.1, 1.0)`.
6. The window parameter table includes `x_target`, `x0`, and `k0` when those columns are available.
7. The notebook handles missing diagnostic columns gracefully.
8. No Python MiNES scheduler or workflow file is modified.
