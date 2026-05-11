# ClaudeCode instruction: Notebook-only MiNES rescue diagnostics update

## Scope

Modify only this file:

```text
notebooks/mines_variance_fusion_visualization.ipynb
```

Do **not** modify:

```text
scripts/mines_variance_fusion.py
```

Do **not** change the MiNES Python scheduler, rescue algorithm, GT rescue logic, budget logic, NEQ scheduling, or PMF reconstruction. This task is notebook-only. The purpose is to improve diagnostics so I can inspect how the current GT rescue formula uses `k0`, `x0`, `s`, `m_s`, and `sigma_s` before deciding whether to change the scheduler later.

Also keep the previous rule:

```python
# Do not call plt.close() anywhere in this notebook.
```

Remove any existing `plt.close()` calls if they are present.

---

## 1. Make RMSE and variance plots log-scale

For all plots whose y-axis is RMSE or variance-like, set:

```python
ax.set_yscale("log")
```

This applies to plots using columns such as:

```text
rmse_bestfit
global_variance
max_global_variance
mean_global_variance
median_global_variance
target_variance
variance_before
variance_after
best_variance_value
```

Do **not** apply log scale to PMF values.

Before plotting on log scale, mask non-positive and non-finite values. Add a helper near the plotting helper section:

```python
def positive_for_log(values):
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0.0] = np.nan
    return arr
```

Use it whenever plotting variance/RMSE on a log y-axis.

Example:

```python
ax.plot(df["used_ksteps"], positive_for_log(df["rmse_bestfit"]), marker="o")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
```

For variance profiles:

```python
ax.plot(global_pmf_df["x"], positive_for_log(global_pmf_df["global_variance"]))
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
```

---

## 2. Simplify the window-parameter table

In the main window-parameter table, do **not** display these columns:

```text
x_most
mean_minus_center_x
mean_minus_x_most
```

Keep the essential columns only:

```text
window or name
side
generation
x_m or center_x
mean_x
std_x
k
q_next
target_source
```

Use robust column selection so the notebook does not crash if a column is absent:

```python
def display_existing_columns(df, columns, max_rows=None):
    cols = [c for c in columns if c in df.columns]
    if not cols:
        print("No requested columns found.")
        return
    out = df[cols].copy()
    if max_rows is not None:
        out = out.tail(max_rows)
    display(out)
```

For the window table:

```python
window_cols = [
    "window", "name", "side", "generation",
    "x_m", "center_x", "mean_x", "std_x",
    "k", "q_next", "target_source",
]
display_existing_columns(windows_df, window_cols)
```

If both `window` and `name` exist, prefer `window`. If only `name` exists, show `name`.

---

## 3. Add a new section: Rescue GT diagnostics

Add a markdown heading:

```markdown
## Rescue GT diagnostics
```

This section should load and display `rescue_summary.csv` if it exists. It should not fail if the file is missing.

Use the same result-root logic already used in the notebook. The file is expected at:

```text
<system_root>/MINES/<label>/raw/seed_<seed>/rescue_summary.csv
```

Example:

```python
rescue_summary_path = result_root / "rescue_summary.csv"
if rescue_summary_path.exists():
    rescue_df = pd.read_csv(rescue_summary_path)
else:
    rescue_df = pd.DataFrame()
    print(f"Missing rescue_summary.csv: {rescue_summary_path}")
```

---

## 4. Compute derived GT rescue quantities in the notebook

Some GT quantities may already be present in `rescue_summary.csv`, but some may need to be derived. Do not modify the scheduler to write new columns. Compute derived columns in the notebook.

Add this helper:

```python
def add_rescue_gt_derived_columns(rescue_df):
    rescue_df = rescue_df.copy()

    def numeric_col(name):
        if name in rescue_df.columns:
            return pd.to_numeric(rescue_df[name], errors="coerce")
        return pd.Series(np.nan, index=rescue_df.index, dtype=float)

    s = numeric_col("gt_s_eff")
    valid_s = np.isfinite(s) & (s > 0.0) & (s < 1.0)

    sigma_L = numeric_col("gt_sigma_L")
    sigma_R = numeric_col("gt_sigma_R")
    x0_L = numeric_col("gt_x0_L")
    x0_R = numeric_col("gt_x0_R")
    k0_L = numeric_col("gt_k0_L")
    k0_R = numeric_col("gt_k0_R")

    rescue_df["gt_sigma_s"] = np.where(
        valid_s,
        (1.0 - s) * sigma_L + s * sigma_R,
        np.nan,
    )

    rescue_df["gt_x0_s"] = np.where(
        valid_s,
        (1.0 - s) * x0_L + s * x0_R,
        np.nan,
    )

    rescue_df["gt_k0_s"] = np.where(
        valid_s,
        (1.0 - s) * k0_L + s * k0_R,
        np.nan,
    )

    if "x_rescue_target" in rescue_df.columns:
        rescue_df["gt_m_s"] = pd.to_numeric(rescue_df["x_rescue_target"], errors="coerce")
    elif "target_bin_x" in rescue_df.columns:
        rescue_df["gt_m_s"] = pd.to_numeric(rescue_df["target_bin_x"], errors="coerce")
    else:
        rescue_df["gt_m_s"] = np.nan

    return rescue_df
```

Apply it after loading:

```python
if not rescue_df.empty:
    rescue_df = add_rescue_gt_derived_columns(rescue_df)
```

Interpretation of the derived quantities:

```text
gt_s_eff    = interpolation coordinate between the two GT endpoint umbrellas
gt_m_s      = target mean used by GT rescue; currently ms = x_rescue_target
gt_sigma_s  = interpolated target standard deviation
gt_x0_s     = interpolated background harmonic center
gt_k0_s     = interpolated background harmonic stiffness/curvature
```

---

## 5. Add a detailed Rescue GT table

Display a detailed diagnostic table with all available columns from this list:

```python
rescue_gt_detail_cols = [
    "round",
    "target_priority",
    "target_reason",
    "x_rescue_target",
    "target_bin_x",
    "target_variance",
    "rescue_center_x",
    "rescue_k",
    "rescue_center_rule",
    "rescue_k_rule",

    "gt_left_cluster",
    "gt_right_cluster",
    "gt_left_boundary",
    "gt_right_boundary",
    "gt_boundary_pair_reason",

    "gt_left_center_x",
    "gt_right_center_x",
    "gt_left_mean_x",
    "gt_right_mean_x",
    "gt_left_std_x",
    "gt_right_std_x",

    "gt_x0_L",
    "gt_k0_L",
    "gt_x0_R",
    "gt_k0_R",

    "gt_m_L",
    "gt_m_R",
    "gt_sigma_L",
    "gt_sigma_R",
    "gt_s_eff",
    "gt_m_s",
    "gt_sigma_s",
    "gt_x0_s",
    "gt_k0_s",
    "gt_x_raw",
    "gt_k_raw",
    "gt_used_midpoint_fallback",
    "gt_fallback_reason",
]
display_existing_columns(rescue_df, rescue_gt_detail_cols)
```

This table is meant to show which endpoint umbrellas were used and how GT constructed the rescue `x_m` and `k_m`.

---

## 6. Add a compact rescue debug table

Add a second, compact table that is easier to read:

```python
rescue_gt_compact_cols = [
    "round",
    "x_rescue_target",
    "target_bin_x",
    "target_variance",
    "gt_left_boundary",
    "gt_right_boundary",
    "gt_s_eff",
    "gt_m_s",
    "gt_sigma_s",
    "gt_x0_s",
    "gt_k0_s",
    "rescue_center_x",
    "rescue_k",
    "gt_x_raw",
    "gt_k_raw",
    "gt_used_midpoint_fallback",
    "gt_fallback_reason",
]
display_existing_columns(rescue_df, rescue_gt_compact_cols)
```

This compact table is the main table I will use to debug whether `x0_s` and `k0_s` are causing unexpected rescue centers/stiffness.

---

## 7. Add optional GT diagnostic plots

If the required columns exist, add two plots.

### Plot A: rescue target versus rescue center

Plot against `round` if available, otherwise use row index.

Required y-columns:

```text
x_rescue_target
rescue_center_x
```

Example:

```python
if not rescue_df.empty and {"x_rescue_target", "rescue_center_x"}.issubset(rescue_df.columns):
    x_axis = rescue_df["round"] if "round" in rescue_df.columns else np.arange(len(rescue_df))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, pd.to_numeric(rescue_df["x_rescue_target"], errors="coerce"), marker="o", label="x_rescue_target")
    ax.plot(x_axis, pd.to_numeric(rescue_df["rescue_center_x"], errors="coerce"), marker="s", label="rescue_center_x")
    ax.set_xlabel("rescue round")
    ax.set_ylabel("x")
    ax.set_title("Rescue target vs designed rescue center")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
```

### Plot B: GT stiffness decomposition

Plot against `round` if available, otherwise use row index.

Y-columns, if available:

```text
rescue_k
gt_k_raw
gt_k0_s
```

Use log y-scale. For `gt_k0_s`, plot only positive values using `positive_for_log`. If `gt_k0_s` is negative, do not plot it on log scale, but optionally print a note that negative values were masked.

Example:

```python
stiffness_cols = [c for c in ["rescue_k", "gt_k_raw", "gt_k0_s"] if c in rescue_df.columns]
if not rescue_df.empty and stiffness_cols:
    x_axis = rescue_df["round"] if "round" in rescue_df.columns else np.arange(len(rescue_df))
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in stiffness_cols:
        ax.plot(x_axis, positive_for_log(rescue_df[col]), marker="o", label=col)
    ax.set_yscale("log")
    ax.set_xlabel("rescue round")
    ax.set_ylabel("stiffness / curvature")
    ax.set_title("GT rescue stiffness decomposition")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()
```

Do not call `plt.close()`.

---

## 8. Keep existing notebook visual style improvements

Retain the previous notebook improvements:

1. Larger title and label fonts.
2. Grid lines on every plot.
3. Finer x-ticks using something like:

```python
ax.set_xticks(np.arange(x_min, x_max + 0.1, 1.0))
```

where appropriate for x-coordinate plots.

---

## 9. Acceptance criteria

The task is complete if:

1. Only `notebooks/mines_variance_fusion_visualization.ipynb` is modified.
2. `scripts/mines_variance_fusion.py` is unchanged.
3. No `plt.close()` call exists in the notebook.
4. RMSE and variance plots use log-scale y axes.
5. The main window table no longer displays:
   - `x_most`
   - `mean_minus_center_x`
   - `mean_minus_x_most`
6. A new `Rescue GT diagnostics` section exists.
7. The Rescue GT diagnostics section shows, when available:
   - `gt_x0_L`, `gt_k0_L`, `gt_x0_R`, `gt_k0_R`
   - `gt_s_eff`
   - derived `gt_m_s`, `gt_sigma_s`, `gt_x0_s`, `gt_k0_s`
   - `gt_left_boundary`, `gt_right_boundary`
   - `rescue_center_x`, `rescue_k`, `gt_x_raw`, `gt_k_raw`
8. Missing files or missing columns are handled gracefully without crashing the notebook.
9. The notebook can be run top-to-bottom after these changes.
