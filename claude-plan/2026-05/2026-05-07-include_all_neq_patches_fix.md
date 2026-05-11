# ClaudeCode Task: Preserve and Include All Valid NEQ/MTS Patches in MiNES Hybrid PMF

## Context

The current MiNES variance-fusion workflow appears to lose useful NEQ information after EQ windows are reclustered. In `--pmf-method hybrid`, the code currently uses the `patches` list rebuilt during each call to `reconstruct_chain()`. Because `reconstruct_chain()` rebuilds clusters and then only loops over **current adjacent EQ clusters**, older NEQ segments can disappear from the current global fit when they become internal to a merged EQ cluster or no longer match the current selected boundary pair.

This is not the desired behavior. In MiNES, every successfully generated bidirectional NEQ segment that can produce a valid `NEQ_MTS` patch should remain usable evidence for the global inverse-variance fusion, even if later EQ clustering changes the current neighbor graph.

## Main goal

Modify the MiNES code so that **all valid NEQ/MTS patches ever generated are retained and included in the global PMF fit**, especially in `--pmf-method hybrid` and `--pmf-method neq`.

The intended behavior is:

```text
hybrid PMF = current EQ cluster patches + all valid NEQ_MTS patches ever generated
neq PMF    = all valid NEQ_MTS patches ever generated
eq PMF     = EQ PMF with variance logic as currently defined, but do not delete NEQ patches from disk or diagnostics
```

Do not allow EQ clustering or reclustering to silently discard previously valid NEQ evidence.

---

## Problem to fix

Currently, the code pattern is approximately:

```python
clusters, js_rows = build_eq_clusters(...)
segments = []
patches = []

for cluster in clusters:
    patches.append(build_eq_cluster_patch(...))

for cluster_idx in range(len(clusters) - 1):
    left_cluster = clusters[cluster_idx]
    right_cluster = clusters[cluster_idx + 1]
    segment = get_or_run_segment_for_current_neighbor_pair(...)
    segments.append(segment)
    neq_patch = build_neq_mts_patch(segment, ...)
    patches.append(neq_patch)

if pmf_method == "hybrid":
    patches_for_global = list(patches)
```

This means `patches` contains only:

1. EQ patches from the current cluster graph.
2. NEQ patches corresponding to current neighboring cluster pairs.

It does **not** necessarily contain older valid NEQ patches, such as `SEG_L0__R0`, `SEG_R0__R1`, `SEG_L1__R1`, etc., if these no longer match the current adjacent-cluster graph.

This is wrong for variance-fusion MiNES.

---

## Required implementation

### 1. Add persistent NEQ patch storage

Create a persistent dictionary outside `reconstruct_chain()` and pass it into `reconstruct_chain()`:

```python
neq_patch_store: dict[str, PMFPatch] = {}
neq_patch_status: dict[str, dict[str, Any]] = {}
```

Key by segment name, for example:

```python
neq_patch_store[segment.name] = patch
```

or by boundary-pair key:

```python
neq_patch_store[(segment.left_boundary.name, segment.right_boundary.name)] = patch
```

Prefer `segment.name` for CSV readability, but include left/right boundary names in diagnostics.

### 2. Store every successfully built NEQ patch

Whenever `build_neq_mts_patch()` succeeds, store the patch persistently:

```python
neq_patch = build_neq_mts_patch(...)
neq_patch_store[segment.name] = neq_patch
neq_patch_status[segment.name] = {
    "segment": segment.name,
    "left_boundary": segment.left_boundary.name,
    "right_boundary": segment.right_boundary.name,
    "mts_patch_built": 1,
    "included_in_global_fit": 1,
    "reason": "valid_neq_mts_patch_persisted",
    ...
}
```

If the same segment is rebuilt later, overwrite the stored patch only if the new patch is valid. Do not remove a previously valid patch merely because the segment is no longer a current neighbor edge.

### 3. Do not rebuild old NEQ patches unnecessarily

If a segment already has a valid stored NEQ patch, do not repeat expensive bootstrapping unless explicitly needed. Reuse the existing patch:

```python
if segment.name in neq_patch_store:
    neq_patch = neq_patch_store[segment.name]
else:
    neq_patch = build_neq_mts_patch(...)
    neq_patch_store[segment.name] = neq_patch
```

This will also reduce runtime.

### 4. Change patch selection for global PMF

At the end of `reconstruct_chain()`, split patch types explicitly:

```python
eq_patches_current = [p for p in patches if p.kind == "EQ_MBAR"]
neq_patches_all = list(neq_patch_store.values())
```

Then set global patch lists as:

```python
if pmf_method == "neq":
    patches_for_global = neq_patches_all
elif pmf_method == "hybrid":
    patches_for_global = eq_patches_current + neq_patches_all
elif pmf_method == "eq":
    patches_for_global = build_eq_pmf_with_neq_variance(
        eq_patches=eq_patches_current,
        neq_patches=neq_patches_all,
        grid=grid,
    )
```

Important: for `hybrid`, do **not** use only the local `patches` list. Use all persistent NEQ patches.

### 5. Avoid duplicate NEQ patches

Make sure a patch is included only once in the global fit. If the current reconstruction also appends the newly built NEQ patch to `patches`, avoid double inclusion by constructing `patches_for_global` from:

```python
eq_patches_current + unique(list(neq_patch_store.values()))
```

Do not append stored NEQ patches twice.

### 6. Keep current-neighbor diagnostics separate from inclusion diagnostics

Add two distinct concepts:

```text
is_current_neighbor_edge: whether this segment connects the currently adjacent EQ clusters
included_in_global_fit: whether this segment's NEQ_MTS patch is included in the global PMF
```

A segment can have:

```text
is_current_neighbor_edge = 0
included_in_global_fit = 1
reason = old_valid_neq_patch_retained
```

This is the desired behavior.

---

## Required output diagnostics

### 1. Add `all_neq_patches.csv`

Write this table at the main output root and inside each snapshot/state directory if snapshots are used.

Suggested columns:

```text
segment
left_boundary
right_boundary
patch_name
patch_kind
n_forward
n_reverse
n_covered_bins
is_current_neighbor_edge
is_internal_to_current_eq_cluster
is_long_range_or_obsolete_edge
mts_patch_built
included_in_global_fit
patch_root
pmf_file
variance_file
summary_file
reason
```

Definitions:

- `is_current_neighbor_edge = 1` if the segment is between the current chosen adjacent-cluster boundary pair.
- `is_internal_to_current_eq_cluster = 1` if both boundary windows currently belong to the same EQ cluster.
- `is_long_range_or_obsolete_edge = 1` if both endpoints exist but the segment is not a current neighbor edge and not internal.
- `included_in_global_fit = 1` if the valid NEQ patch is included in `patches_for_global` for `neq` or `hybrid` mode.

### 2. Extend `patches.csv`

Make sure `patches.csv` reflects the actual patches used in the global fit, or add a new file:

```text
patches_used_for_global_fit.csv
```

This file should include both:

```text
current EQ_MBAR patches
all retained NEQ_MTS patches
```

Suggested columns:

```text
name
kind
source_names
n_covered_bins
included_in_global_fit
inclusion_source
patch_root
pmf_file
variance_file
aligned_pmf_file
summary_file
```

where `inclusion_source` is one of:

```text
current_eq_cluster_patch
current_neighbor_neq_patch
retained_old_neq_patch
```

### 3. Add a notebook diagnostic plot

Update `notebooks/mines_variance_fusion_visualization.ipynb` to show a table or plot of retained NEQ patches:

```text
segment vs n_covered_bins
color/marker by current_neighbor vs retained_old
```

Also print a warning if there are valid NEQ patches on disk that are not included in the global fit.

---

## Additional helper functions to implement

### 1. Current cluster membership map

Add a helper:

```python
def build_window_to_cluster_map(clusters: list[EQCluster]) -> dict[str, str]:
    mapping = {}
    for cluster in clusters:
        for w in cluster.windows:
            mapping[w.name] = cluster.name
    return mapping
```

### 2. Current neighbor segment key set

Add a helper:

```python
def current_neighbor_segment_keys(segments: list[NEQSegment]) -> set[str]:
    return {segment.name for segment in segments}
```

or use boundary-pair keys:

```python
def segment_boundary_key(segment: NEQSegment) -> tuple[str, str]:
    return (segment.left_boundary.name, segment.right_boundary.name)
```

### 3. Classify retained NEQ segment

Add:

```python
def classify_neq_segment_against_current_clusters(
    segment: NEQSegment,
    clusters: list[EQCluster],
    current_neighbor_names: set[str],
) -> dict[str, Any]:
    window_to_cluster = build_window_to_cluster_map(clusters)
    left_cluster = window_to_cluster.get(segment.left_boundary.name, "")
    right_cluster = window_to_cluster.get(segment.right_boundary.name, "")

    is_current = segment.name in current_neighbor_names
    is_internal = bool(left_cluster and right_cluster and left_cluster == right_cluster)
    is_long_range = bool(left_cluster and right_cluster and left_cluster != right_cluster and not is_current)

    return {
        "left_current_cluster": left_cluster,
        "right_current_cluster": right_cluster,
        "is_current_neighbor_edge": int(is_current),
        "is_internal_to_current_eq_cluster": int(is_internal),
        "is_long_range_or_obsolete_edge": int(is_long_range),
    }
```

---

## Expected behavior after the fix

For example, suppose the workflow generated these NEQ segments:

```text
SEG_L0__R0
SEG_L1__R1
SEG_L2__R2
SEG_R2__rescue_01
```

Later, after EQ clustering, the current cluster graph may only require:

```text
SEG_L2__R2
SEG_R2__rescue_01
```

The global hybrid PMF should still include all valid patches:

```text
EQ_MBAR patches from current clusters
NEQ_MTS patch from SEG_L0__R0
NEQ_MTS patch from SEG_L1__R1
NEQ_MTS patch from SEG_L2__R2
NEQ_MTS patch from SEG_R2__rescue_01
```

Older NEQ patches may receive low weight where their variance is high, but they should not be silently discarded.

---

## Important constraints

1. Do not hard-stitch the PMF.
2. Keep the final PMF as inverse-variance weighted global fusion with fitted patch offsets.
3. Do not remove EQ clustering. EQ clustering should still determine current EQ MBAR patches.
4. Do not let EQ clustering delete or hide valid NEQ evidence.
5. Do not include invalid NEQ patches with no finite coverage.
6. Avoid duplicate inclusion of the same NEQ patch.
7. Preserve existing output files where possible, but add the new diagnostics.
8. Keep the code auditable from disk outputs.

---

## Acceptance criteria

The fix is correct if:

1. `--pmf-method hybrid` includes current EQ patches plus all valid NEQ patches ever generated.
2. `--pmf-method neq` includes all valid NEQ patches ever generated.
3. Older NEQ patches remain included even after EQ windows merge into larger clusters.
4. `all_neq_patches.csv` clearly shows whether each NEQ patch is current, internal, obsolete/long-range, and included.
5. `patches_used_for_global_fit.csv` lists the actual patches used in the global fit.
6. The global PMF coverage no longer decreases merely because a previously valid NEQ segment is no longer a current adjacent-cluster edge.
7. No NEQ patch is included twice.
8. Runtime does not unnecessarily repeat NEQ patch bootstrapping for segments that already have valid stored patches.

---

## Suggested minimal code-change strategy

The least invasive approach is:

1. Add `neq_patch_store` and `neq_patch_status` in `main()` beside `segment_store`.
2. Pass these dictionaries into `reconstruct_chain()`.
3. Inside `reconstruct_chain()`, when an NEQ patch is built successfully, save it to `neq_patch_store`.
4. At global patch selection, use `list(neq_patch_store.values())` instead of only current NEQ patches.
5. Add diagnostics writer functions for `all_neq_patches.csv` and `patches_used_for_global_fit.csv`.
6. Update the notebook to visualize retained NEQ patches.

