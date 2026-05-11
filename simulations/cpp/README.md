# C++ Nonequilibrium Simulation

This folder contains the standalone C++ driver for EQ, NEQ, adaptive umbrella
sampling (AUS), and well-tempered metadynamics runs.

The helper shell scripts now live in `simulations/`. The C++ source, binary,
design note, and outputs remain in `simulations/cpp/`.

The active code path is now isotropic-only for the harmonic control:

- one scalar stiffness `k`
- optional midpoint scaling `k_midscale`
- no anisotropic `kt/ko`
- no ellipse angle `theta`

The previous anisotropic implementation is preserved in
`simulations/cpp/anisotropic_backup/` and `src/cpp/anisotropic_backup/`.

It also supports a 1D log-sum-exp double-well potential for runs with
`-one-dimension x` or `-one-dimension y`:

```text
U(q) = -kT * ln(exp(-(k0 * (q - x0)^2) / kT) + exp(-(k1 * (q - x1)^2) / kT - E1))
```

where `q` is the active coordinate and `beta = 1 / kT`.

You can override the simulation thermal `kT`, timestep, and friction from the
command line with `-thermal_kT <value>`, `-dt <value>`, and `-gamma <value>`.
These affect the Langevin dynamics, and `-thermal_kT` also affects
`Double-well_1D`.

## Build

From the repo root:

```bash
clang++ -O2 -std=c++17 simulations/cpp/neq_sim.cpp -o simulations/cpp/neq_sim
```

## Command-line usage

EQ sampling:

```bash
./simulations/cpp/neq_sim -pot "Muller-Brown" -center_xy -0.6,1.4 -eq_out EQ_lamb0.csv -k 200 -T_eq 1000 -eq_nout 1000 -out_dir ./simulations/cpp/outputs/Muller-Brown
./simulations/cpp/neq_sim -pot "Muller-Brown" -center_xy 0.6,0.0 -eq_out EQ_lamb1.csv -k 200 -T_eq 1000 -out_dir ./simulations/cpp/outputs/Muller-Brown
```

1D EQ sampling with the double-well potential:

```bash
./simulations/cpp/neq_sim -pot "Double-well_1D" -one-dimension x -thermal_kT 1.0 -center_xy -1.0,0.0 -eq_out EQ_lamb0.csv -k 200 -k0 8 -x0 -1.0 -k1 8 -x1 1.0 -E1 0.0 -T_eq 1000 -out_dir ./simulations/cpp/outputs/Double-well_1D
./simulations/cpp/neq_sim -pot "Double-well_1D" -one-dimension x -thermal_kT 1.0 -center_xy 1.0,0.0 -eq_out EQ_lamb1.csv -k 200 -k0 8 -x0 -1.0 -k1 8 -x1 1.0 -E1 0.0 -T_eq 1000 -out_dir ./simulations/cpp/outputs/Double-well_1D
```

NEQ simulation without a path file:

```bash
./simulations/cpp/neq_sim -pot "Muller-Brown" -eq0 EQ_lamb0.csv -eq1 EQ_lamb1.csv -N_neq 100 -T_neq 100 -neq_nout 50 -A_center -0.6,1.4 -B_center 0.6,0.0 -k 200 -k_midscale 3 -out_dir ./simulations/cpp/outputs/Muller-Brown
```

NEQ simulation with a path file:

```bash
./simulations/cpp/neq_sim -pot "Muller-Brown" -eq0 EQ_lamb0.csv -eq1 EQ_lamb1.csv -fpath EQ_iter.csv -N_neq 100 -T_neq 100 -out_dir ./simulations/cpp/outputs/Muller-Brown
```

1D adaptive umbrella sampling:

```bash
./simulations/cpp/neq_sim -pot "Double-well_1D" -one-dimension x -thermal_kT 1.0 -k0 8 -x0 -1.0 -k1 8 -x1 1.0 -E1 0.0 -aus_mode -T_aus 4000 -aus_nout 400 -aus_k 80 -aus_spacing 0.25 -aus_max_windows 30 -aus_min_count 10 -aus_overlap_cutoff 0.05 -aus_fes_out aus_fes.csv -out_dir ./simulations/cpp/outputs/Double-well_1D/AUS
```

2D adaptive umbrella sampling:

```bash
./simulations/cpp/neq_sim -pot "Muller-Brown" -aus_mode -A_center -0.6,1.4 -B_center 0.6,0.0 -T_aus 4000 -aus_nout 400 -aus_k 150 -aus_spacing 0.20 -aus_max_windows 40 -aus_min_count 10 -aus_overlap_cutoff 0.05 -aus_fes_out aus_fes.csv -out_dir ./simulations/cpp/outputs/Muller-Brown/AUS
```

1D well-tempered metadynamics:

```bash
./simulations/cpp/neq_sim -pot "Double-well_1D" -one-dimension x -thermal_kT 1.0 -k0 8 -x0 -1.0 -k1 8 -x1 1.0 -E1 0.0 -meta_start_xy -1.0,0.0 -T_meta 40000 -meta_out meta_traj.csv -meta_hills_out meta_hills.csv -meta_fes_out meta_fes.csv -meta_w0 0.6 -meta_sigma_x 0.08 -meta_biasfactor 10 -meta_stride 200 -meta_nout 1000 -out_dir ./simulations/cpp/outputs/Double-well_1D/WTMeta
```

2D well-tempered metadynamics:

```bash
./simulations/cpp/neq_sim -pot "Muller-Brown" -meta_start_xy -0.6,1.4 -T_meta 40000 -meta_out meta_traj.csv -meta_hills_out meta_hills.csv -meta_fes_out meta_fes.csv -meta_w0 0.6 -meta_sigma_x 0.08 -meta_sigma_y 0.08 -meta_biasfactor 10 -meta_stride 200 -meta_nout 1000 -out_dir ./simulations/cpp/outputs/Muller-Brown/WTMeta
```

Standalone metadynamics FES reconstruction:

```bash
./simulations/cpp/neq_sim -pot "Muller-Brown" -meta_fes_mode -meta_hills_in ./simulations/cpp/outputs/Muller-Brown/WTMeta/meta_hills.csv -meta_fes_out meta_fes.csv -meta_biasfactor 10 -meta_nx 120 -meta_ny 120 -out_dir ./simulations/cpp/outputs/Muller-Brown/WTMetaAnalysis
```

## Path input/output

- If `-fpath` is not provided, a linear path between `A` and `B` is used.
- If `-fpath` is provided, the file must exist.
- The actual path used is written to `neq_path.csv` in the output directory, or to `-path_out` if provided.

Accepted path file formats:

- `x,y`
- `lambda,x,y`
- `lambda,x,y,k`

The active isotropic code also tolerates old anisotropic path CSV rows with
extra columns by reading the first stiffness column and ignoring the rest.

The output `neq_path.csv` contains:

```text
lambda,x0,y0,k
```

## Output files

NEQ trajectories for each `i` in `[0, N_neq-1]`:

- `neq_fwd_i.csv`
- `neq_bwd_i.csv`

Each NEQ CSV has columns:

```text
step,lambda,x,y,base_u,bias_u,work
```

Adaptive umbrella outputs:

- `aus_window_<id>.csv`
- `aus_windows.csv`
- `aus_schedule.csv`
- `aus_fes.csv`

Each umbrella window trajectory has columns:

```text
step,x,y,base_u,bias_u
```

Metadynamics outputs:

- `meta_traj.csv`
- `meta_hills.csv`
- `meta_fes.csv` when requested

## Notes

- Supported potentials: `Muller-Brown`, `Six-hump_camel`, `Three_wells`, `Double-well_1D`.
- `Double-well_1D` requires `-one-dimension x` or `-one-dimension y`.
- Use `-thermal_kT`, `-dt`, and `-gamma` to override the thermal `kT`, timestep, and friction used by the integrator.
- The 1D potential parameters default to `k0=10`, `x0=-1`, `k1=10`, `x1=1`, `E1=0`.
- Adaptive umbrella uses harmonic windows and grows to neighboring centers only
  when the parent window has enough support in that direction.
- Metadynamics supports both 1D direct CVs (`x` or `y`) and the full 2D CV `(x, y)`.
- The metadynamics FES reconstruction uses the standard well-tempered estimate
  `F(s) ~= -(gamma / (gamma - 1)) * V_bias(s) + C`.
- Default parameters are defined in `src/cpp/sim_config.h`.
