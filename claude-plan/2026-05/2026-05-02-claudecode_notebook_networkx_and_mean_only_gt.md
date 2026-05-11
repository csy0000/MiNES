# Claude Code Task: Add NetworkX uncertainty graph and simplify GT PMF visualization

You are modifying the Jupyter notebook:

```text
notebooks/mines_variance_fusion_visualization.ipynb
```

or the current uploaded/active notebook with the same content.

## Goal

Make two notebook-side visualization changes:

1. Add a new cell that uses `networkx` to visualize the uncertainty network between all EQ windows and NEQ-connected windows.
2. Simplify the GT visualization: only show the **mean-only GT-derived PMF**, the **global estimated PMF**, and the **analytical PMF**. Do **not** show any old `s`-dependent harmonic curves anymore.

---

# 1. Add a NetworkX uncertainty graph cell

Add a new notebook section, preferably after the window/segment/patch summary cells and before the final PMF comparison plots.

The section title should be:

```markdown
## EQ/NEQ uncertainty network
```

## Purpose

The graph should visualize the uncertainty as a distance between EQ windows.

Each node is an EQ window. Edges represent either:

- EQ-cluster/BAR connections between neighboring windows in the same EQ cluster.
- NEQ/CFT connections between windows or clusters connected by NEQ trajectories.

Use **all available EQ and NEQ data** that the notebook can find. Do not restrict the plot to only the final selected PMF patches if more EQ/NEQ connectivity is available in the summary files.

---

## Required visual encoding

### Nodes

Each node should correspond to an EQ window.

Node attributes:

- `name`
- `center_x`
- `mean_x`
- `generation`
- `side`
- `k`

Node color:

- Color nodes by `mean_x`.
- Use a continuous colormap from `xmin` to `xmax`.
- `xmin` should map to blue.
- `xmax` should map to red.
- Prefer `cmap="coolwarm"` or equivalent.

If `mean_x` is unavailable, fall back in this order:

1. `mean`
2. `x_mean`
3. `x_most`
4. `center_x`

### Edges

There are two edge types.

#### EQ/BAR edges

These connect EQ windows that are clustered together.

Draw them as:

```python
style = "solid"
color = "black"
```

The edge uncertainty should be `ddF_BAR`.

Sources for `ddF_BAR`, in priority order:

1. If `clusters.csv`, `patches.csv`, or another summary file already contains a BAR uncertainty column, use it.
   Accept possible column names:
   - `ddF`
   - `delta_f_unc`
   - `bar_delta_f_unc`
   - `BAR_ddF`
   - `bar_uncertainty`
2. If no stored uncertainty exists, try to compute BAR uncertainty using the available EQ tail samples for the two windows.
3. If computation is not possible, still draw the edge, but set:
   ```python
   ddF = np.nan
   uncertainty_source = "missing"
   ```

#### NEQ/CFT edges

These connect EQ windows or clusters connected by NEQ trajectories.

Draw them as:

```python
style = "dashed"
color = "black"
```

The edge uncertainty should be `ddF_CFT`.

Sources for `ddF_CFT`, in priority order:

1. Use stored CFT/BAR/Crooks uncertainty columns from `segments.csv`, `patches.csv`, or NEQ summary files.
   Accept possible column names:
   - `cft_delta_f_unc`
   - `cft_ddF`
   - `crooks_delta_f_unc`
   - `crooks_ddF`
   - `delta_f_unc`
   - `ddF`
2. If stored uncertainty is unavailable, try to compute it from forward/reverse work values using the existing bidirectional estimator utilities if they are already imported in the notebook/project.
3. If computation is not possible, still draw the edge with:
   ```python
   ddF = np.nan
   uncertainty_source = "missing"
   ```

---

## Edge distance / layout

Use the uncertainty as an effective graph distance.

Implementation suggestion:

```python
edge_weight = 1.0 / max(ddF, ddF_floor)
```

for layout weighting, because NetworkX spring layout treats larger weights as stronger/shorter springs.

Use:

```python
ddF_floor = 1e-6
```

If `ddF` is missing or non-finite, use a weak default weight, e.g.:

```python
edge_weight = 0.1
```

The graph layout should therefore place high-uncertainty edges farther apart and low-uncertainty edges closer together.

Recommended layout:

```python
pos = nx.spring_layout(G, weight="layout_weight", seed=7, k=None)
```

Optionally, initialize x-positions by `mean_x` or `center_x` if easy, but this is not required.

---

## Plot requirements

The cell should produce:

1. A NetworkX graph plot.
2. A small edge summary table.

The graph should have:

- Node labels as window names.
- Node colors from blue to red according to `mean_x`.
- Solid black edges for EQ/BAR.
- Dashed black edges for NEQ/CFT.
- Edge labels showing:
  ```text
  BAR ddF=...
  ```
  or
  ```text
  CFT ddF=...
  ```

Use a colorbar labeled:

```text
mean x
```

Use a legend for:

- solid black = EQ/BAR
- dashed black = NEQ/CFT

The edge summary table should contain at least:

```text
left, right, kind, ddF, uncertainty_source
```

where `kind` is either:

```text
EQ_BAR
NEQ_CFT
```

---

## Robustness requirements

The cell must not crash if optional files or uncertainty columns are absent.

Use robust file loading from the existing notebook variables, such as:

```python
seed_root
windows_df
clusters_df
segments_df
patches_df
```

If a DataFrame is missing, load it from:

```python
seed_root / "windows.csv"
seed_root / "clusters.csv"
seed_root / "segments.csv"
seed_root / "patches.csv"
seed_root / "rescue_summary.csv"
```

If not enough information exists to build some edge type, print a warning but still plot whatever graph can be constructed.

Example warning style:

```python
print("[network] No NEQ/CFT edges could be inferred from segments.csv or patches.csv")
```

Do not hide missing information silently.

---

## Suggested implementation structure

Use helper functions inside the notebook cell:

```python
def first_existing_col(df, candidates):
    ...

def get_window_position(row):
    ...

def add_window_nodes(G, windows_df):
    ...

def add_eq_bar_edges(G, windows_df, clusters_df, patches_df):
    ...

def add_neq_cft_edges(G, windows_df, segments_df, patches_df):
    ...

def draw_uncertainty_network(G):
    ...
```

The code does not need to be overly abstract, but it should be readable.

---

# 2. Simplify the GT PMF visualization

Find the current GT visualization cell in the notebook.

It likely plots multiple curves derived from different `s` values, for example:

- `s`-dependent harmonic PMFs
- curves derived from different `s`
- old harmonic reconstructions where `k0` and `x0` depend on `s`

Remove or disable all of those old `s`-dependent harmonic plots.

## New required plot

The GT/PMF comparison plot should show only these curves:

1. Analytical PMF
2. Global estimated PMF from the MiNES protocol
3. Mean-only GT-derived PMF

Optional but useful:

4. Global PMF uncertainty band if `global_variance` is present

The title should be something like:

```text
Mean-only GT PMF vs global MiNES PMF
```

The legend should include only:

```text
Analytical PMF
Global MiNES PMF
Mean-only GT PMF
```

and optionally:

```text
Global ±1σ
```

Do not include any labels containing:

```text
s =
s-dependent
harmonic by s
k0(s)
x0(s)
```

---

## Mean-only GT meaning

The mean-only GT curve should be based on the current scheme:

- Infer the local/background harmonic relation using only the EQ means.
- `k0` and `x0` are no longer `s`-dependent.
- The interpolation coordinate `s` may be used internally only to target the transported mean, but the plot should not show a family of curves over `s`.

In other words, the notebook should not visualize multiple harmonic PMFs parameterized by `s`.

The plotted mean-only GT PMF should be a single curve.

If the notebook already has a function that computes the mean-only GT PMF, use it.

If the function does not exist yet, create a small helper in the notebook, e.g.:

```python
def compute_mean_only_gt_pmf(grid, left_mean, right_mean, left_k, right_k, ctx):
    ...
```

but do not over-engineer it. The notebook is for diagnosis/visualization.

---

## Data source preference

Use these data sources where available:

- `global_pmf.csv` for:
  - `x`
  - `global_pmf`
  - `global_variance`
  - `analytic_pmf`
- `windows.csv` or `rescue_summary.csv` for:
  - EQ means
  - `k0`
  - `x0`
  - rescue GT metadata
- Any existing GT diagnostic CSV/JSON if already written by the Python workflow.

The cell should be robust if some GT metadata is missing. In that case, print:

```python
print("[GT] Mean-only GT metadata not found; skipping GT curve.")
```

but still plot the analytical PMF and global MiNES PMF.

---

# 3. Acceptance criteria

The notebook modification is correct if:

1. A new NetworkX uncertainty graph section exists.
2. The graph uses all available EQ windows as nodes.
3. Node color encodes mean `x` from `xmin` blue to `xmax` red.
4. EQ/BAR edges are solid black.
5. NEQ/CFT edges are dashed black.
6. Edge labels show uncertainty `ddF` when available.
7. The edge summary table is displayed.
8. The graph cell does not crash when some uncertainties are missing.
9. The GT visualization no longer shows old `s`-dependent harmonic curves.
10. The GT visualization shows only:
    - analytical PMF,
    - global MiNES estimated PMF,
    - one mean-only GT PMF curve if metadata is available.
11. The legend does not contain old `s`-dependent labels.

---

# 4. Important note

Do not change the simulation workflow in this task. This is a notebook-only visualization update.

Do not change the final PMF estimator here.

Do not remove the raw summary-loading cells unless necessary.

Only add the network diagnostic and simplify the GT plotting cell.
