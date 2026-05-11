# Task: Add bounded GT and linear NEQ protocol modes to MiNES

You are working on:

- `scripts/mines_variance_fusion.py`
- `scripts/run_mines_variance_fusion.sh`

The current MiNES code constructs NEQ segments using a single constant `protocol_k` and linear center interpolation. Replace this with two explicit bounded protocol modes:

1. `GT`, default: local Gaussian-transport-inspired protocol.
2. `linear`: direct interpolation of umbrella center and sqrt-stiffness.

Do **not** add a fallback mode. Both modes must always apply:

- `k_s ∈ [k_min, k_max]`
- `x_s ∈ [min(x_L, x_R), max(x_L, x_R)]`
- exact thermodynamic endpoints

The goal is to make every NEQ segment auditable and to ensure the MTS reconstruction reads the exact protocol actually used.

---

## 1. Add CLI argument

In `parse_args()`, add:

```python
parser.add_argument(
    "--neq-protocol-mode",
    choices=["GT", "linear", "gt"],
    default="GT",
    help="NEQ bridge protocol mode. GT is local Gaussian transport; linear interpolates x and sqrt(k).",
)
```

Normalize after parsing or inside protocol construction:

```python
protocol_mode = str(args.neq_protocol_mode).upper()
```

Treat `"gt"` and `"GT"` as the same mode.

---

## 2. Add shell wrapper support

In `scripts/run_mines_variance_fusion.sh`, add default:

```bash
NEQ_PROTOCOL_MODE="GT"
```

Add parser case:

```bash
--neq-protocol-mode)
  NEQ_PROTOCOL_MODE="$2"
  shift 2
  ;;
```

Add to `CMD`:

```bash
--neq-protocol-mode "${NEQ_PROTOCOL_MODE}"
```

Because the script already collects unknown options into `EXTRA_ARGS`, this is not strictly required for functionality, but add it for discoverability and reproducibility.

---

## 3. Extend `NEQSegment`

Add protocol metadata fields to the `NEQSegment` dataclass:

```python
protocol_mode: str = "GT"
protocol_metadata: dict[str, Any] = field(default_factory=dict)
protocol_k_min: float | None = None
protocol_k_max: float | None = None
protocol_x_min: float | None = None
protocol_x_max: float | None = None
protocol_clip_fraction_k: float = 0.0
protocol_clip_fraction_x: float = 0.0
```

Keep `protocol_k` for backward compatibility, but redefine it as a summary value, e.g. mean protocol stiffness:

```python
protocol_k = float(np.nanmean(forward_ks))
```

Do not rely on it as the actual path stiffness.

---

## 4. Add helper functions

Add these functions near the existing `pair_protocol_k()` / `linear_path_centers()` helpers.

### 4.1 Safe sample statistics

```python
def window_tail_mean_sigma(window: EnsembleWindow) -> tuple[float, float]:
    x = eq_tail_samples(window)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise RuntimeError(f"Need at least two finite tail samples for {window.name}.")
    mean = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(f"Invalid tail sigma for {window.name}: {sigma}")
    return mean, sigma
```

### 4.2 Generic clipping helper

```python
def clip_with_flag(value: float, lower: float, upper: float) -> tuple[float, bool]:
    clipped = float(min(max(float(value), float(lower)), float(upper)))
    was_clipped = bool(abs(clipped - float(value)) > 1.0e-12)
    return clipped, was_clipped
```

### 4.3 Linear protocol builder

```python
def build_linear_bridge_protocol(
    *,
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    n_time: int,
    k_min: float,
    k_max: float,
) -> dict[str, Any]:
    if n_time <= 1:
        n_time = 2

    xL = float(left_window.center_x)
    xR = float(right_window.center_x)
    kL = float(left_window.k)
    kR = float(right_window.k)

    x_low = min(xL, xR)
    x_high = max(xL, xR)

    s_values = np.linspace(0.0, 1.0, num=int(n_time))
    centers: list[float] = []
    ks: list[float] = []
    rows: list[dict[str, Any]] = []
    n_clip_x = 0
    n_clip_k = 0

    for idx, s in enumerate(s_values):
        if idx == 0:
            x_s = xL
            k_s = kL
            raw_x = x_s
            raw_k = k_s
            clipped_x = False
            clipped_k = False
        elif idx == len(s_values) - 1:
            x_s = xR
            k_s = kR
            raw_x = x_s
            raw_k = k_s
            clipped_x = False
            clipped_k = False
        else:
            raw_x = (1.0 - float(s)) * xL + float(s) * xR
            sqrt_k = (1.0 - float(s)) * math.sqrt(kL) + float(s) * math.sqrt(kR)
            raw_k = sqrt_k * sqrt_k

            x_s, clipped_x = clip_with_flag(raw_x, x_low, x_high)
            k_s, clipped_k = clip_with_flag(raw_k, k_min, k_max)

        n_clip_x += int(clipped_x)
        n_clip_k += int(clipped_k)
        centers.append(float(x_s))
        ks.append(float(k_s))
        rows.append(
            {
                "step_index": int(idx),
                "s": float(s),
                "mode": "linear",
                "x_raw": float(raw_x),
                "k_raw": float(raw_k),
                "x": float(x_s),
                "k": float(k_s),
                "x_clipped": int(clipped_x),
                "k_clipped": int(clipped_k),
                "m_target": "",
                "sigma_target": "",
                "m_actual": "",
                "k0_local": "",
                "q_local": "",
            }
        )

    return {
        "centers": centers,
        "ks": ks,
        "rows": rows,
        "metadata": {
            "protocol_mode": "linear",
            "x_min": float(x_low),
            "x_max": float(x_high),
            "k_min": float(k_min),
            "k_max": float(k_max),
            "clip_fraction_x": float(n_clip_x) / float(len(s_values)),
            "clip_fraction_k": float(n_clip_k) / float(len(s_values)),
        },
    }
```

### 4.4 GT protocol builder

Use the local formulas:

\[
K_L = 1/\sigma_L^2,\qquad K_R = 1/\sigma_R^2
\]

\[
k_0 = \frac{1}{2}\left[(K_L-k_L)+(K_R-k_R)\right]
\]

\[
q = \frac{1}{2}\left[(K_Lm_L-k_Lx_L)+(K_Rm_R-k_Rx_R)\right]
\]

For interior points:

\[
m_s=(1-s)m_L+s m_R
\]

\[
\sigma_s=(1-s)\sigma_L+s\sigma_R
\]

\[
K_s=1/\sigma_s^2
\]

\[
k_s=\operatorname{clip}(K_s-k_0,k_{\min},k_{\max})
\]

\[
x_s=\operatorname{clip}\left(\frac{(k_0+k_s)m_s-q}{k_s},x_{\min},x_{\max}\right)
\]

Also store the actual mean after clipping:

\[
m_s^{actual}=\frac{q+k_sx_s}{k_0+k_s}
\]

Implementation:

```python
def build_gt_bridge_protocol(
    *,
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    n_time: int,
    k_min: float,
    k_max: float,
) -> dict[str, Any]:
    if n_time <= 1:
        n_time = 2

    xL = float(left_window.center_x)
    xR = float(right_window.center_x)
    kL = float(left_window.k)
    kR = float(right_window.k)

    x_low = min(xL, xR)
    x_high = max(xL, xR)

    mL, sigmaL = window_tail_mean_sigma(left_window)
    mR, sigmaR = window_tail_mean_sigma(right_window)

    KL = 1.0 / (sigmaL * sigmaL)
    KR = 1.0 / (sigmaR * sigmaR)

    k0_local = 0.5 * ((KL - kL) + (KR - kR))
    q_local = 0.5 * ((KL * mL - kL * xL) + (KR * mR - kR * xR))

    s_values = np.linspace(0.0, 1.0, num=int(n_time))
    centers: list[float] = []
    ks: list[float] = []
    rows: list[dict[str, Any]] = []
    n_clip_x = 0
    n_clip_k = 0

    for idx, s in enumerate(s_values):
        if idx == 0:
            x_s = xL
            k_s = kL
            raw_x = x_s
            raw_k = k_s
            clipped_x = False
            clipped_k = False
            m_target = mL
            sigma_target = sigmaL
            K_target = KL
            m_actual = mL
        elif idx == len(s_values) - 1:
            x_s = xR
            k_s = kR
            raw_x = x_s
            raw_k = k_s
            clipped_x = False
            clipped_k = False
            m_target = mR
            sigma_target = sigmaR
            K_target = KR
            m_actual = mR
        else:
            m_target = (1.0 - float(s)) * mL + float(s) * mR
            sigma_target = (1.0 - float(s)) * sigmaL + float(s) * sigmaR
            K_target = 1.0 / (sigma_target * sigma_target)

            raw_k = K_target - k0_local
            k_s, clipped_k = clip_with_flag(raw_k, k_min, k_max)

            raw_x = ((k0_local + k_s) * m_target - q_local) / k_s
            x_s, clipped_x = clip_with_flag(raw_x, x_low, x_high)

            denom = k0_local + k_s
            m_actual = (
                (q_local + k_s * x_s) / denom
                if math.isfinite(denom) and abs(denom) > 1.0e-12
                else float("nan")
            )

        n_clip_x += int(clipped_x)
        n_clip_k += int(clipped_k)
        centers.append(float(x_s))
        ks.append(float(k_s))
        rows.append(
            {
                "step_index": int(idx),
                "s": float(s),
                "mode": "GT",
                "x_raw": float(raw_x),
                "k_raw": float(raw_k),
                "x": float(x_s),
                "k": float(k_s),
                "x_clipped": int(clipped_x),
                "k_clipped": int(clipped_k),
                "m_target": float(m_target),
                "sigma_target": float(sigma_target),
                "K_target": float(K_target),
                "m_actual": float(m_actual),
                "mean_tracking_error": (
                    float(m_actual - m_target)
                    if math.isfinite(float(m_actual)) and math.isfinite(float(m_target))
                    else ""
                ),
                "k0_local": float(k0_local),
                "q_local": float(q_local),
                "mL": float(mL),
                "sigmaL": float(sigmaL),
                "mR": float(mR),
                "sigmaR": float(sigmaR),
                "KL": float(KL),
                "KR": float(KR),
            }
        )

    return {
        "centers": centers,
        "ks": ks,
        "rows": rows,
        "metadata": {
            "protocol_mode": "GT",
            "x_min": float(x_low),
            "x_max": float(x_high),
            "k_min": float(k_min),
            "k_max": float(k_max),
            "clip_fraction_x": float(n_clip_x) / float(len(s_values)),
            "clip_fraction_k": float(n_clip_k) / float(len(s_values)),
            "mL": float(mL),
            "sigmaL": float(sigmaL),
            "mR": float(mR),
            "sigmaR": float(sigmaR),
            "KL": float(KL),
            "KR": float(KR),
            "k0_local": float(k0_local),
            "q_local": float(q_local),
        },
    }
```

### 4.5 Dispatcher

```python
def build_bridge_protocol(
    *,
    mode: str,
    left_window: EnsembleWindow,
    right_window: EnsembleWindow,
    n_time: int,
    k_min: float,
    k_max: float,
) -> dict[str, Any]:
    normalized_mode = str(mode).upper()
    if normalized_mode == "GT":
        return build_gt_bridge_protocol(
            left_window=left_window,
            right_window=right_window,
            n_time=n_time,
            k_min=k_min,
            k_max=k_max,
        )
    if normalized_mode == "LINEAR":
        return build_linear_bridge_protocol(
            left_window=left_window,
            right_window=right_window,
            n_time=n_time,
            k_min=k_min,
            k_max=k_max,
        )
    raise ValueError(f"Unknown NEQ protocol mode: {mode}")
```

---

## 5. Update `run_neq_protocol`

Add parameter:

```python
neq_protocol_mode: str,
```

to `run_neq_protocol`.

Replace the current constant-`protocol_k` path logic.

### Important implementation detail

The current code calls `run_neq_edge(...)` before writing `forward_path.csv` and `reverse_path.csv`. That helper appears to generate trajectories internally from only `left_center`, `right_center`, and a constant `k`.

For this task, the actual simulation must use the generated time-dependent protocol, not merely write it afterward for analysis. Therefore:

1. Check whether the simulator wrapper supports explicit protocol path files.
2. If it already has a helper for protocol-file NEQ, use it.
3. If not, add a new helper in this script or in the local simulator interface, but do not modify unrelated AUS logic.

The final behavior must be:

- build forward protocol arrays before running NEQ
- write `forward_path.csv`
- write `reverse_path.csv`
- run forward trajectories using `forward_path.csv`
- run reverse trajectories using `reverse_path.csv`
- MTS reads the same path files that were used by the simulator

Do not leave the code in a state where the binary ran a constant-k linear protocol while the analysis reads a GT protocol. That would invalidate the NEQ work and MTS reconstruction.

### Replacement structure

Inside `run_neq_protocol`:

```python
n_time_requested = int(max(t_neq, 1))

forward_protocol = build_bridge_protocol(
    mode=neq_protocol_mode,
    left_window=boundary_left,
    right_window=boundary_right,
    n_time=n_time_requested,
    k_min=float(k_min),
    k_max=float(k_max),
)

reverse_protocol = build_bridge_protocol(
    mode=neq_protocol_mode,
    left_window=boundary_right,
    right_window=boundary_left,
    n_time=n_time_requested,
    k_min=float(k_min),
    k_max=float(k_max),
)

forward_path_file = protocol_root / "forward_path.csv"
reverse_path_file = protocol_root / "reverse_path.csv"

write_protocol_path(forward_path_file, forward_protocol["centers"], forward_protocol["ks"])
write_protocol_path(reverse_path_file, reverse_protocol["centers"], reverse_protocol["ks"])

write_csv(
    protocol_root / "forward_protocol_diagnostics.csv",
    ordered_fieldnames(forward_protocol["rows"]),
    forward_protocol["rows"],
)
write_csv(
    protocol_root / "reverse_protocol_diagnostics.csv",
    ordered_fieldnames(reverse_protocol["rows"]),
    reverse_protocol["rows"],
)
write_json(
    protocol_root / "protocol_summary.json",
    {
        "mode": str(forward_protocol["metadata"]["protocol_mode"]),
        "forward": forward_protocol["metadata"],
        "reverse": reverse_protocol["metadata"],
    },
)
```

Then run the simulator using those path files.

If the existing helper `run_neq_edge` cannot accept protocol path files, add a new helper, for example:

```python
run_neq_edge_with_protocol_paths(
    bin_path=bin_path,
    ctx=ctx,
    eq_left=boundary_left.eq_file,
    eq_right=boundary_right.eq_file,
    forward_protocol_path=forward_path_file,
    reverse_protocol_path=reverse_path_file,
    n_traj_per_direction=int(n_neq_traj),
    t_neq=int(t_neq),
    nout=int(max(t_neq, 1)),
    seed=int(seed),
    out_dir=root,
)
```

Use whatever low-level binary flags are already supported by `neq_sim` for protocol files. If the binary does not currently support protocol-file switching, make the smallest necessary simulator-interface change so that NEQ switching reads a CSV with columns:

```text
step,x0,k
```

or whatever `write_protocol_path_raw()` already writes. Keep the format consistent with `write_protocol_path()`.

After running, keep the existing logic:

```python
normalize_flat_neq_outputs_to_segment_layout(root)
forward_files, reverse_files = discover_neq_trajectories(root)
forward_trajectories = [read_csv_rows(path) for path in forward_files]
reverse_trajectories = [read_csv_rows(path) for path in reverse_files]
```

Then compute:

```python
n_time = min(len(rows) for rows in forward_trajectories + reverse_trajectories)
```

If `n_time` is shorter than the requested protocol length, trim the written path files or rewrite them to exactly match `n_time`:

```python
forward_centers = forward_protocol["centers"][:n_time]
forward_ks = forward_protocol["ks"][:n_time]
reverse_centers = reverse_protocol["centers"][:n_time]
reverse_ks = reverse_protocol["ks"][:n_time]

write_protocol_path(forward_path_file, forward_centers, forward_ks)
write_protocol_path(reverse_path_file, reverse_centers, reverse_ks)
```

The files used by `build_neq_mts_patch()` must match the time dimension of the trajectory arrays.

---

## 6. Update segment summary

In the `NEQSegment(...)` construction, set:

```python
protocol_mode=str(forward_protocol["metadata"]["protocol_mode"]),
protocol_metadata={
    "forward": forward_protocol["metadata"],
    "reverse": reverse_protocol["metadata"],
},
protocol_k=float(np.nanmean(np.asarray(forward_ks, dtype=float))),
protocol_k_min=float(np.nanmin(np.asarray(forward_ks, dtype=float))),
protocol_k_max=float(np.nanmax(np.asarray(forward_ks, dtype=float))),
protocol_x_min=float(np.nanmin(np.asarray(forward_centers, dtype=float))),
protocol_x_max=float(np.nanmax(np.asarray(forward_centers, dtype=float))),
protocol_clip_fraction_k=float(forward_protocol["metadata"]["clip_fraction_k"]),
protocol_clip_fraction_x=float(forward_protocol["metadata"]["clip_fraction_x"]),
```

In `segment_summary.json`, add:

```python
"protocol_mode": segment.protocol_mode,
"protocol_k_mean": segment.protocol_k,
"protocol_k_min_observed": segment.protocol_k_min,
"protocol_k_max_observed": segment.protocol_k_max,
"protocol_x_min_observed": segment.protocol_x_min,
"protocol_x_max_observed": segment.protocol_x_max,
"protocol_clip_fraction_k": segment.protocol_clip_fraction_k,
"protocol_clip_fraction_x": segment.protocol_clip_fraction_x,
"protocol_summary_file": relative_to_root(protocol_root / "protocol_summary.json", out_root),
"forward_protocol_diagnostics_file": relative_to_root(protocol_root / "forward_protocol_diagnostics.csv", out_root),
"reverse_protocol_diagnostics_file": relative_to_root(protocol_root / "reverse_protocol_diagnostics.csv", out_root),
```

Keep the existing `"protocol_k"` field but make it mean stiffness for backward compatibility.

---

## 7. Update segment table rows

In `build_segment_rows()`, add columns:

```python
"protocol_mode": segment.protocol_mode,
"protocol_k_mean": segment.protocol_k,
"protocol_k_min_observed": segment.protocol_k_min,
"protocol_k_max_observed": segment.protocol_k_max,
"protocol_x_min_observed": segment.protocol_x_min,
"protocol_x_max_observed": segment.protocol_x_max,
"protocol_clip_fraction_k": segment.protocol_clip_fraction_k,
"protocol_clip_fraction_x": segment.protocol_clip_fraction_x,
"protocol_summary_file": relative_to_root(segment.root / "protocols" / "protocol_summary.json", out_root),
"forward_protocol_diagnostics_file": relative_to_root(segment.root / "protocols" / "forward_protocol_diagnostics.csv", out_root),
"reverse_protocol_diagnostics_file": relative_to_root(segment.root / "protocols" / "reverse_protocol_diagnostics.csv", out_root),
```

---

## 8. Thread the argument through all calls

Every call to `run_neq_protocol(...)` must pass:

```python
neq_protocol_mode=str(args.neq_protocol_mode),
```

There are calls in:

- the growth stage
- `reconstruct_chain(...)`, for missing neighboring-cluster segments
- rescue reconstruction through `reconstruct_chain(...)`

Make sure `reconstruct_chain()` does not need a separate parameter because it already receives `args`.

---

## 9. Include protocol mode in run summary

In `run_request.json`, the existing `parameters` dump will include `neq_protocol_mode` automatically after the CLI argument is added.

Also add to `mines_variance_fusion_summary.json`:

```python
"neq_protocol_mode": str(args.neq_protocol_mode).upper(),
```

---

## 10. Acceptance criteria

The patch is correct if:

1. `--neq-protocol-mode GT` is the default.
2. `--neq-protocol-mode linear` is supported.
3. Linear mode uses:

   \[
   x_s=(1-s)x_L+s x_R
   \]

   \[
   k_s=\operatorname{clip}\left(((1-s)\sqrt{k_L}+s\sqrt{k_R})^2,k_{\min},k_{\max}\right)
   \]

4. GT mode uses:

   \[
   k_0=\frac12[(1/\sigma_L^2-k_L)+(1/\sigma_R^2-k_R)]
   \]

   \[
   q=\frac12[(m_L/\sigma_L^2-k_Lx_L)+(m_R/\sigma_R^2-k_Rx_R)]
   \]

   \[
   m_s=(1-s)m_L+s m_R
   \]

   \[
   \sigma_s=(1-s)\sigma_L+s\sigma_R
   \]

   \[
   k_s=\operatorname{clip}(1/\sigma_s^2-k_0,k_{\min},k_{\max})
   \]

   \[
   x_s=\operatorname{clip}(((k_0+k_s)m_s-q)/k_s,\min(x_L,x_R),\max(x_L,x_R))
   \]

5. Both modes enforce exact endpoint states:

   \[
   (x_0,k_0)_{\mathrm{protocol}}=(x_L,k_L)
   \]

   \[
   (x_1,k_1)_{\mathrm{protocol}}=(x_R,k_R)
   \]

6. The actual simulator uses the same protocol path files that MTS later reads.
7. Every segment writes:
   - `protocols/forward_path.csv`
   - `protocols/reverse_path.csv`
   - `protocols/forward_protocol_diagnostics.csv`
   - `protocols/reverse_protocol_diagnostics.csv`
   - `protocols/protocol_summary.json`
8. `segments.csv` contains protocol mode, observed min/max x and k, and clipping fractions.
9. The old constant-`protocol_k` behavior is removed except as a backward-compatible summary field.
10. `bash scripts/run_mines_variance_fusion.sh --neq-protocol-mode linear --quick-test` runs.
11. `bash scripts/run_mines_variance_fusion.sh --neq-protocol-mode GT --quick-test` runs.
12. `build_neq_mts_patch()` reads the exact protocol used by the simulation.

---

## Critical warning

Do not merely write GT path files after running the old constant-k NEQ simulation.

`build_neq_mts_patch()` reads `forward_path.csv` and `reverse_path.csv`, so those files must describe the actual switching protocol used by the simulator. Otherwise, the work values and MTS PMF reconstruction are inconsistent.
