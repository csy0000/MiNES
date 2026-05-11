#!/usr/bin/env bash
set -euo pipefail

# This script runs the active 1D DoubleWell benchmark workflow.
# - US is a fixed-window umbrella screen over k at fixed dx=0.5 with no directional split.
# - AUS grows one paired left/right adaptive umbrella chain from both
#   endpoints.
# - NES is screened over k and uses only the bidirectional estimator:
#   50 forward + 50 backward switching trajectories per target budget.
# - MINES is an adaptive milestone-chain NEQ method grown inward from both endpoints.
# - WT-MTD uses two walkers, one from each basin.
#
# Active storage policy:
# - run one raw method block
# - analyze it immediately
# - keep processed PMF outputs and a reduced 1000-sample trajectory file
# - delete the larger raw trajectory files before moving on
#
# Main parameter blocks:
# - RUN_US / RUN_AUS / RUN_NES / RUN_MINES / RUN_MTD: set to 0 to skip a method family.
# - POT_*: underlying 1D double-well definition.
# - THERMAL_KT / DT / GAMMA: Langevin dynamics settings.
# - GRID_*: x-grid used in downstream PMF analysis.
# - US_*: screened US spring constants at fixed dx=0.5.
# - AUS_*: paired adaptive umbrella controls for the quantile-based
#   window-growth rule, screening both parent-only 4-term polynomial and
#   parent-only cubic-spline slope estimation, then using iterative post-match
#   interested-region ESS-guided rescue / redistribution under the shared
#   fixed total budget.
# - NES_*: screened NES pulling stiffness values and fixed switching controls.
# - MINES_*: screened adaptive milestone pulling stiffness values, explicit
#   frontier dx, and fixed growth controls.
# - MTD_*: screened two-walker metadynamics biasfactors and fixed controls.
# - SEEDS: replicate seeds shared across all methods.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
SIM_DIR="$ROOT_DIR/simulations/cpp"
BIN="$SIM_DIR/neq_sim"
REDUCER="$ROOT_DIR/src/analysis/analysis_US_MTD.py"
PLOTTER="$ROOT_DIR/analysis/notebook/plot_doublewell_benchmark.py"
NOTEBOOK="$ROOT_DIR/analysis/notebook/doublewell_benchmark_results.ipynb"
METHOD_CONTEXT_WRITER="$ROOT_DIR/simulations/write_method_contexts.py"
ADAPTIVE_RUNNER="$ROOT_DIR/simulations/adaptive_methods.py"

RUN_US="${RUN_US:-1}"
RUN_AUS="${RUN_AUS:-1}"
RUN_NES="${RUN_NES:-1}"
RUN_MINES="${RUN_MINES:-1}"
RUN_MTD="${RUN_MTD:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RUN_NOTEBOOK="${RUN_NOTEBOOK:-1}"

SEEDS=(101)

POTENTIAL_NAME="${POTENTIAL_NAME:-Double-well_1D}"
POT_K0="${POT_K0:-0.05}"
POT_X0="${POT_X0:--10.0}"
POT_K1="${POT_K1:-0.05}"
POT_X1="${POT_X1:-10.0}"
POT_E1="${POT_E1:-0.0}"

THERMAL_KT="${THERMAL_KT:-1.0}"
DT="${DT:-0.0005}"
GAMMA="${GAMMA:-1.0}"
GRID_DX="${GRID_DX:-0.1}"
GRID_XMIN="${GRID_XMIN:--12.0}"
GRID_XMAX="${GRID_XMAX:-12.0}"

US_K_VALUES=(5.0 10.0 20.0 50.0)
US_DX_VALUES=(0.5)
US_SAMPLE_STRIDE_STEPS=1000

AUS_QNEXT_VALUES=(0.95)
AUS_ALPHA_VALUES=(3.0)
AUS_DX="${AUS_DX:-0.05}"
AUS_ENDPOINT_K="${AUS_ENDPOINT_K:-1.0}"
AUS_K_MIN_VALUES=(1.0)
AUS_K_MAX_VALUES=(50.0)
AUS_FIT_METHOD_VALUES=(poly_4term_parent cubic_spline_parent)
AUS_EQ_STEPS="${AUS_EQ_STEPS:-100000}"
AUS_EQ_NOUT="${AUS_EQ_NOUT:-1000}"
AUS_ANALYSIS_TAIL_FRACTION="${AUS_ANALYSIS_TAIL_FRACTION:-0.9}"
AUS_DECISION_MAX_SAMPLES_PER_WINDOW="${AUS_DECISION_MAX_SAMPLES_PER_WINDOW:-1000}"
AUS_MAX_ITERATIONS="${AUS_MAX_ITERATIONS:-50}"
AUS_REFINE_METRIC="${AUS_REFINE_METRIC:-interested_region_fraction_of_average_ess}"
AUS_REFINE_HALF_WIDTH_POINTS="${AUS_REFINE_HALF_WIDTH_POINTS:-5}"
AUS_REFINE_K_ADDITION="${AUS_REFINE_K_ADDITION:-10.0}"
AUS_REFINE_RESCUE_K_MAX="${AUS_REFINE_RESCUE_K_MAX:-100.0}"
AUS_REFINE_ESS_MIN_FRACTION="${AUS_REFINE_ESS_MIN_FRACTION:-${AUS_REFINE_ESS_THRESHOLD:-0.05}}"

NES_EQ_STEPS=10000
NES_EQ_NOUT=1000
NES_N_TRAJ_PER_DIRECTION=50
NES_NOUT=1000
NES_K_VALUES=(0.1 1.0 5.0 10.0)
NES_K_MIDSCALE=1.0

MINES_K_PULL_VALUES=(1.0 5.0 10.0)
MINES_DX="${MINES_DX:-0.05}"
MINES_EQ_STEPS="${MINES_EQ_STEPS:-10000}"
MINES_EQ_NOUT="${MINES_EQ_NOUT:-1000}"
MINES_N_TRAJ_PER_DIRECTION="${MINES_N_TRAJ_PER_DIRECTION:-25}"
MINES_T_NEQ="${MINES_T_NEQ:-10000}"
MINES_NEQ_NOUT="${MINES_NEQ_NOUT:-1000}"
MINES_ESS_MIN="${MINES_ESS_MIN:-20}"
MINES_OVERLAP_MIN="${MINES_OVERLAP_MIN:-0.10}"
MINES_WORK_OVERLAP_MIN="${MINES_WORK_OVERLAP_MIN:-0.05}"
MINES_MAX_DEPTH_PER_SIDE="${MINES_MAX_DEPTH_PER_SIDE:-10}"

MTD_W0=4.0
MTD_SIGMA=0.8
MTD_BIASFACTOR_VALUES=(10.0 100.0 1000.0 10000.0)
MTD_STRIDE=1000
MTD_SAMPLE_STRIDE_STEPS=1000

slug_float() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

json_number_array() {
  local out=""
  local value
  for value in "$@"; do
    if [[ -n "$out" ]]; then
      out+=", "
    fi
    out+="$value"
  done
  printf '[%s]' "$out"
}

json_string_array() {
  local out=""
  local value
  for value in "$@"; do
    if [[ -n "$out" ]]; then
      out+=", "
    fi
    out+="\"$value\""
  done
  printf '[%s]' "$out"
}

if [[ -n "${SEEDS_CSV:-}" ]]; then
  IFS=',' read -r -a SEEDS <<<"$SEEDS_CSV"
fi
if [[ -n "${US_K_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a US_K_VALUES <<<"$US_K_VALUES_CSV"
fi
if [[ -n "${AUS_ALPHA_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a AUS_ALPHA_VALUES <<<"$AUS_ALPHA_VALUES_CSV"
fi
if [[ -n "${AUS_FIT_METHOD_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a AUS_FIT_METHOD_VALUES <<<"$AUS_FIT_METHOD_VALUES_CSV"
fi
if [[ -n "${AUS_QNEXT_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a AUS_QNEXT_VALUES <<<"$AUS_QNEXT_VALUES_CSV"
fi
if [[ -n "${AUS_K_MIN_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a AUS_K_MIN_VALUES <<<"$AUS_K_MIN_VALUES_CSV"
fi
if [[ -n "${AUS_K_MAX_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a AUS_K_MAX_VALUES <<<"$AUS_K_MAX_VALUES_CSV"
fi
if [[ -n "${US_DX_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a US_DX_VALUES <<<"$US_DX_VALUES_CSV"
fi
if [[ -n "${NES_K_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a NES_K_VALUES <<<"$NES_K_VALUES_CSV"
fi
if [[ -n "${MINES_K_PULL_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a MINES_K_PULL_VALUES <<<"$MINES_K_PULL_VALUES_CSV"
fi
if [[ -n "${MTD_BIASFACTOR_VALUES_CSV:-}" ]]; then
  IFS=',' read -r -a MTD_BIASFACTOR_VALUES <<<"$MTD_BIASFACTOR_VALUES_CSV"
fi

us_combo_label() {
  local k="$1"
  local dx="$2"
  printf 'k_%s__dx_%s' "$(slug_float "$k")" "$(slug_float "$dx")"
}

aus_combo_label() {
  local qnext="$1"
  local alpha="$2"
  local fit_method="$3"
  local kmin="$4"
  local kmax="$5"
  printf 'qnext_%s__alpha_%s__fit_%s__kmin_%s__kmax_%s' \
    "$(slug_float "$qnext")" \
    "$(slug_float "$alpha")" \
    "$fit_method" \
    "$(slug_float "$kmin")" \
    "$(slug_float "$kmax")"
}

nes_combo_label() {
  local k="$1"
  printf 'k_%s' "$(slug_float "$k")"
}

mines_combo_label() {
  local k="$1"
  printf 'k_pull_%s' "$(slug_float "$k")"
}

mtd_combo_label() {
  local biasfactor="$1"
  printf 'biasfactor_%s' "$(slug_float "$biasfactor")"
}

if [[ -n "${TIME_STEPS_CSV:-}" ]]; then
  IFS=',' read -r -a TIME_STEPS <<<"$TIME_STEPS_CSV"
else
  TIME_STEPS=()
  while IFS= read -r value; do
    TIME_STEPS+=("$value")
  done < <(python3 - <<'PY'
import numpy as np
values = np.geomspace(1.0e4, 1.0e7, 21)
steps = [int(round(v / 100.0) * 100.0) for v in values]
if len(set(steps)) != len(steps):
    raise SystemExit("Rounded time grid produced duplicate values.")
for value in steps:
    print(value)
PY
  )
fi

TIME_LABELS=()
for total_steps in "${TIME_STEPS[@]}"; do
  TIME_LABELS+=("T_${total_steps}")
done

MAX_TOTAL_STEPS="${TIME_STEPS[$((${#TIME_STEPS[@]} - 1))]}"
MTD_PER_WALKER_STEPS=$((MAX_TOTAL_STEPS / 2))
MTD_NOUT=$((MTD_PER_WALKER_STEPS / MTD_SAMPLE_STRIDE_STEPS))

LEFT_X=$POT_X0
RIGHT_X=$POT_X1

SYSTEM_SLUG=$(
  printf 'DoubleWell__k0_%s__x0_%s__k1_%s__x1_%s__E1_%s__kT_%s__dt_%s__gamma_%s' \
    "$(slug_float "$POT_K0")" \
    "$(slug_float "$POT_X0")" \
    "$(slug_float "$POT_K1")" \
    "$(slug_float "$POT_X1")" \
    "$(slug_float "$POT_E1")" \
    "$(slug_float "$THERMAL_KT")" \
    "$(slug_float "$DT")" \
    "$(slug_float "$GAMMA")"
)
SYSTEM_ROOT="$ROOT_DIR/data/1D/$SYSTEM_SLUG"

COMMON_ARGS=(
  -pot "$POTENTIAL_NAME"
  -one-dimension x
  -thermal_kT "$THERMAL_KT"
  -dt "$DT"
  -gamma "$GAMMA"
  -k0 "$POT_K0" -x0 "$POT_X0" -k1 "$POT_K1" -x1 "$POT_X1" -E1 "$POT_E1"
)

us_window_info() {
  python3 - "$LEFT_X" "$RIGHT_X" "$1" "$MAX_TOTAL_STEPS" "$US_SAMPLE_STRIDE_STEPS" <<'PY'
import math
import sys

left = float(sys.argv[1])
right = float(sys.argv[2])
spacing = float(sys.argv[3])
total_steps = int(sys.argv[4])
sample_stride = int(sys.argv[5])
distance = abs(right - left)
n_intervals = int(round(distance / spacing))
if n_intervals <= 0:
    n_intervals = 1
reconstructed = n_intervals * spacing
if abs(reconstructed - distance) > 1e-6 * max(1.0, distance):
    n_intervals = int(math.ceil(distance / spacing))
n_windows = n_intervals + 1
base = total_steps // n_windows
remainder = total_steps % n_windows
max_steps = base + (1 if remainder > 0 else 0)
nout = max(1, int(math.ceil(max_steps / float(sample_stride))))
print(f"{n_windows} {base} {remainder} {nout}")
PY
}

mkdir -p "$SYSTEM_ROOT"
clang++ -O2 -std=c++17 "$SIM_DIR/neq_sim.cpp" -o "$BIN"

seed_json=$(json_number_array "${SEEDS[@]}")
time_json=$(json_number_array "${TIME_STEPS[@]}")
time_label_json=$(json_string_array "${TIME_LABELS[@]}")
us_k_json=$(json_number_array "${US_K_VALUES[@]}")
us_dx_json=$(json_number_array "${US_DX_VALUES[@]}")
aus_qnext_json=$(json_number_array "${AUS_QNEXT_VALUES[@]}")
aus_alpha_json=$(json_number_array "${AUS_ALPHA_VALUES[@]}")
aus_fit_method_json=$(json_string_array "${AUS_FIT_METHOD_VALUES[@]}")
aus_kmin_json=$(json_number_array "${AUS_K_MIN_VALUES[@]}")
aus_kmax_json=$(json_number_array "${AUS_K_MAX_VALUES[@]}")
nes_k_json=$(json_number_array "${NES_K_VALUES[@]}")
mines_k_json=$(json_number_array "${MINES_K_PULL_VALUES[@]}")
mtd_biasfactor_json=$(json_number_array "${MTD_BIASFACTOR_VALUES[@]}")

US_COMBO_JSON=""
US_COMBO_LABELS=()
US_COMBO_K=()
US_COMBO_DX=()
US_COMBO_N_WINDOWS=()
US_COMBO_NOUT=()
for k in "${US_K_VALUES[@]}"; do
  for dx in "${US_DX_VALUES[@]}"; do
    label=$(us_combo_label "$k" "$dx")
    read -r n_windows steps_base remainder nout <<<"$(us_window_info "$dx")"
    US_COMBO_LABELS+=("$label")
    US_COMBO_K+=("$k")
    US_COMBO_DX+=("$dx")
    US_COMBO_N_WINDOWS+=("$n_windows")
    US_COMBO_NOUT+=("$nout")
    if [[ -n "$US_COMBO_JSON" ]]; then
      US_COMBO_JSON+=", "
    fi
    US_COMBO_JSON+="{\"label\":\"$label\",\"k\":$k,\"dx\":$dx,\"n_windows\":$n_windows,\"steps_per_window\":$steps_base,\"remainder_windows\":$remainder,\"total_steps\":$MAX_TOTAL_STEPS,\"sample_stride_steps\":$US_SAMPLE_STRIDE_STEPS,\"output_samples\":$nout}"
  done
done

AUS_COMBO_JSON=""
AUS_COMBO_LABELS=()
AUS_COMBO_ALPHA=()
for qnext in "${AUS_QNEXT_VALUES[@]}"; do
  for alpha in "${AUS_ALPHA_VALUES[@]}"; do
    for fit_method in "${AUS_FIT_METHOD_VALUES[@]}"; do
      for kmin in "${AUS_K_MIN_VALUES[@]}"; do
        for kmax in "${AUS_K_MAX_VALUES[@]}"; do
          label=$(aus_combo_label "$qnext" "$alpha" "$fit_method" "$kmin" "$kmax")
          AUS_COMBO_LABELS+=("$label")
          AUS_COMBO_ALPHA+=("$alpha")
          if [[ -n "$AUS_COMBO_JSON" ]]; then
            AUS_COMBO_JSON+=", "
          fi
          AUS_COMBO_JSON+="{\"label\":\"$label\",\"q_next\":$qnext,\"alpha\":$alpha,\"fit_method\":\"$fit_method\",\"k_min\":$kmin,\"k_max\":$kmax}"
        done
      done
    done
  done
done

NES_COMBO_JSON=""
NES_COMBO_LABELS=()
NES_COMBO_K=()
for k in "${NES_K_VALUES[@]}"; do
  label=$(nes_combo_label "$k")
  NES_COMBO_LABELS+=("$label")
  NES_COMBO_K+=("$k")
  if [[ -n "$NES_COMBO_JSON" ]]; then
    NES_COMBO_JSON+=", "
  fi
  NES_COMBO_JSON+="{\"label\":\"$label\",\"k\":$k,\"k_midscale\":$NES_K_MIDSCALE}"
done

MINES_COMBO_JSON=""
MINES_COMBO_LABELS=()
MINES_COMBO_K_PULL=()
for k_pull in "${MINES_K_PULL_VALUES[@]}"; do
  label=$(mines_combo_label "$k_pull")
  MINES_COMBO_LABELS+=("$label")
  MINES_COMBO_K_PULL+=("$k_pull")
  if [[ -n "$MINES_COMBO_JSON" ]]; then
    MINES_COMBO_JSON+=", "
  fi
  MINES_COMBO_JSON+="{\"label\":\"$label\",\"k_pull\":$k_pull}"
done

MTD_COMBO_JSON=""
MTD_COMBO_LABELS=()
MTD_COMBO_BIASFACTOR=()
for biasfactor in "${MTD_BIASFACTOR_VALUES[@]}"; do
  label=$(mtd_combo_label "$biasfactor")
  MTD_COMBO_LABELS+=("$label")
  MTD_COMBO_BIASFACTOR+=("$biasfactor")
  if [[ -n "$MTD_COMBO_JSON" ]]; then
    MTD_COMBO_JSON+=", "
  fi
  MTD_COMBO_JSON+="{\"label\":\"$label\",\"biasfactor\":$biasfactor,\"total_steps\":$MAX_TOTAL_STEPS,\"per_walker_steps\":$MTD_PER_WALKER_STEPS,\"sample_stride_steps\":$MTD_SAMPLE_STRIDE_STEPS,\"meta_nout\":$MTD_NOUT,\"w0\":$MTD_W0,\"sigma\":$MTD_SIGMA,\"stride\":$MTD_STRIDE}"
done

cat > "$SYSTEM_ROOT/run_context.json" <<EOF
{
  "system_name": "DoubleWell",
  "system_slug": "$SYSTEM_SLUG",
  "potential_name": "$POTENTIAL_NAME",
  "one_dimension": "x",
  "basins": {
    "left": $LEFT_X,
    "right": $RIGHT_X
  },
  "potential": {
    "k0": $POT_K0,
    "x0": $POT_X0,
    "k1": $POT_K1,
    "x1": $POT_X1,
    "E1": $POT_E1
  },
  "thermal_kT": $THERMAL_KT,
  "dt": $DT,
  "gamma": $GAMMA,
  "grid": {
    "xmin": $GRID_XMIN,
    "xmax": $GRID_XMAX,
    "dx": $GRID_DX
  },
  "seeds": $seed_json,
  "time_grid": {
    "kind": "logspace_steps",
    "t_min": ${TIME_STEPS[0]},
    "t_max": $MAX_TOTAL_STEPS,
    "count": ${#TIME_STEPS[@]},
    "values": $time_json,
    "labels": $time_label_json
  },
  "combo_labeling": {
    "us": "k_<k>__dx_<dx> with decimal points replaced by p",
    "aus": "qnext_<q>__alpha_<a>__fit_<fit_method>__kmin_<kmin>__kmax_<kmax> with decimal points replaced by p",
    "nes": "k_<k> with decimal points replaced by p",
    "mines": "k_pull_<k> with decimal points replaced by p",
    "mtd": "biasfactor_<biasfactor> with decimal points replaced by p"
  },
  "data_layout": {
    "system_root": "data/1D/$SYSTEM_SLUG",
    "run_context": "run_context.json",
    "us_seed_processed": "US/<combo_label>/processed/seed_<seed>.dat",
    "aus_seed_processed": "AUS/<combo_label>/processed/seed_<seed>.dat",
    "nes_seed_processed": "NES/<combo_label>/processed/seed_<seed>.dat",
    "mines_seed_processed": "MINES/<combo_label>/processed/seed_<seed>.dat",
    "mtd_seed_processed": "MTD/<combo_label>/processed/seed_<seed>.dat",
    "benchmark_selected": "benchmark/selected",
    "benchmark_figures": "benchmark/figures",
    "benchmark_gifs": "benchmark/gifs"
  },
  "us_screen": {
    "k_values": $us_k_json,
    "dx_values": $us_dx_json,
    "fixed": {
      "total_steps": $MAX_TOTAL_STEPS,
      "sample_stride_steps": $US_SAMPLE_STRIDE_STEPS
    },
    "combos": [$US_COMBO_JSON]
  },
  "aus_screen": {
    "q_next_values": $aus_qnext_json,
    "alpha_values": $aus_alpha_json,
    "fit_method_values": $aus_fit_method_json,
    "k_min_values": $aus_kmin_json,
    "k_max_values": $aus_kmax_json,
    "fixed": {
      "total_steps": $MAX_TOTAL_STEPS,
      "grid_dx": $AUS_DX,
      "start_x_left": $LEFT_X,
      "start_x_right": $RIGHT_X,
      "endpoint_k": $AUS_ENDPOINT_K,
      "eq_steps": $AUS_EQ_STEPS,
      "eq_nout": $AUS_EQ_NOUT,
      "analysis_tail_fraction": $AUS_ANALYSIS_TAIL_FRACTION,
      "decision_max_samples_per_window": $AUS_DECISION_MAX_SAMPLES_PER_WINDOW,
      "max_iterations": $AUS_MAX_ITERATIONS,
      "refine_metric": "$AUS_REFINE_METRIC",
      "refine_half_width_points": $AUS_REFINE_HALF_WIDTH_POINTS,
      "refine_k_addition": $AUS_REFINE_K_ADDITION,
      "refine_rescue_k_max": $AUS_REFINE_RESCUE_K_MAX,
      "refine_ess_min_fraction": $AUS_REFINE_ESS_MIN_FRACTION,
      "refine_fit_method": "combo_specific",
      "refine_target_rule": "lowest_fractional_ess_bin",
      "refine_extension_policy": "disabled_for_poly_4term_parent__enabled_for_cubic_spline_parent",
      "refine_unresolved_poly_fit_rule": "first_5_left_and_first_5_right__3_term_polynomial",
      "refine_poly_target_coverage_rule": "double_k_until_target_in_qband_or_mark_unresolvable",
      "refine_positive_curvature_rule": "absolute_curvature_plus_k_rescue_at_target_bin",
      "post_match_cycle_mode": "iterative_ess_recheck"
    },
    "combos": [$AUS_COMBO_JSON]
  },
  "nes_screen": {
    "k_values": $nes_k_json,
    "fixed": {
      "eq_steps": $NES_EQ_STEPS,
      "eq_nout": $NES_EQ_NOUT,
      "n_traj_per_direction": $NES_N_TRAJ_PER_DIRECTION,
      "neq_nout": $NES_NOUT,
      "k_midscale": $NES_K_MIDSCALE
    },
    "combos": [$NES_COMBO_JSON]
  },
  "mines_screen": {
    "k_pull_values": $mines_k_json,
    "fixed": {
      "grid_dx": $MINES_DX,
      "eq_steps": $MINES_EQ_STEPS,
      "eq_nout": $MINES_EQ_NOUT,
      "n_traj_per_direction": $MINES_N_TRAJ_PER_DIRECTION,
      "t_neq": $MINES_T_NEQ,
      "neq_nout": $MINES_NEQ_NOUT,
      "ess_min": $MINES_ESS_MIN,
      "overlap_min": $MINES_OVERLAP_MIN,
      "work_overlap_min": $MINES_WORK_OVERLAP_MIN,
      "max_depth_per_side": $MINES_MAX_DEPTH_PER_SIDE
    },
    "combos": [$MINES_COMBO_JSON]
  },
  "mtd_screen": {
    "biasfactor_values": $mtd_biasfactor_json,
    "fixed": {
      "total_steps": $MAX_TOTAL_STEPS,
      "per_walker_steps": $MTD_PER_WALKER_STEPS,
      "sample_stride_steps": $MTD_SAMPLE_STRIDE_STEPS,
      "meta_nout": $MTD_NOUT,
      "w0": $MTD_W0,
      "sigma": $MTD_SIGMA,
      "stride": $MTD_STRIDE
    },
    "combos": [$MTD_COMBO_JSON]
  },
  "rmse_eval_grid": {
    "xmin": -10.0,
    "xmax": 10.0,
    "dx": 0.2
  }
}
EOF

python3 "$METHOD_CONTEXT_WRITER" --system-root "$SYSTEM_ROOT"

run_us_seed() {
  local out_dir="$1"
  local k="$2"
  local dx="$3"
  local seed="$4"
  local nout="$5"
  mkdir -p "$out_dir"
  "$BIN" \
    "${COMMON_ARGS[@]}" \
    -us_mode \
    -T_us_total "$MAX_TOTAL_STEPS" \
    -us_nout "$nout" \
    -us_k "$k" \
    -us_spacing "$dx" \
    -us_seed "$seed" \
    -out_dir "$out_dir" \
    -log "$out_dir/run.log"
}

run_nes_eq() {
  local out_dir="$1"
  local center_xy="$2"
  local eq_out="$3"
  local eq_seed="$4"
  local k="$5"
  mkdir -p "$out_dir"
  "$BIN" \
    "${COMMON_ARGS[@]}" \
    -k "$k" \
    -center_xy "$center_xy" \
    -eq_out "$eq_out" \
    -T_eq "$NES_EQ_STEPS" \
    -eq_nout "$NES_EQ_NOUT" \
    -eq_seed "$eq_seed" \
    -out_dir "$out_dir" \
    -log "$out_dir/${eq_out%.csv}.log"
}

run_nes_neq() {
  local out_dir="$1"
  local eq0="$2"
  local eq1="$3"
  local neq_seed="$4"
  local k="$5"
  local t_neq="$6"
  mkdir -p "$out_dir"
  "$BIN" \
    "${COMMON_ARGS[@]}" \
    -k "$k" \
    -k_midscale "$NES_K_MIDSCALE" \
    -A_center "$LEFT_X,0.0" \
    -B_center "$RIGHT_X,0.0" \
    -eq0 "$eq0" \
    -eq1 "$eq1" \
    -N_neq "$NES_N_TRAJ_PER_DIRECTION" \
    -T_neq "$t_neq" \
    -neq_nout "$NES_NOUT" \
    -neq_seed "$neq_seed" \
    -out_dir "$out_dir" \
    -log "$out_dir/neq.log"
}

run_mtd_walker() {
  local out_dir="$1"
  local start_xy="$2"
  local meta_seed="$3"
  local biasfactor="$4"
  mkdir -p "$out_dir"
  "$BIN" \
    "${COMMON_ARGS[@]}" \
    -meta_start_xy "$start_xy" \
    -T_meta "$MTD_PER_WALKER_STEPS" \
    -meta_out meta_traj.csv \
    -meta_hills_out meta_hills.csv \
    -meta_nout "$MTD_NOUT" \
    -meta_w0 "$MTD_W0" \
    -meta_sigma_x "$MTD_SIGMA" \
    -meta_biasfactor "$biasfactor" \
    -meta_stride "$MTD_STRIDE" \
    -meta_seed "$meta_seed" \
    -out_dir "$out_dir" \
    -log "$out_dir/run.log"
}

if [[ "$RUN_US" != "0" ]]; then
  for ((combo_index = 0; combo_index < ${#US_COMBO_LABELS[@]}; combo_index++)); do
    combo_label="${US_COMBO_LABELS[$combo_index]}"
    k="${US_COMBO_K[$combo_index]}"
    dx="${US_COMBO_DX[$combo_index]}"
    nout="${US_COMBO_NOUT[$combo_index]}"
    for seed in "${SEEDS[@]}"; do
      raw_dir="$SYSTEM_ROOT/US/$combo_label/raw/seed_${seed}"
      run_us_seed "$raw_dir" "$k" "$dx" "$((seed + combo_index * 10000))" "$nout"
      process_args=(process-us-seed --system-root "$SYSTEM_ROOT" --combo-label "$combo_label" --seed "$seed")
      if [[ "$seed" == "${SEEDS[0]}" ]]; then
        process_args+=(--make-gif)
      fi
      python3 "$REDUCER" "${process_args[@]}"
    done
  done
fi

if [[ "$RUN_AUS" != "0" ]]; then
  for ((combo_index = 0; combo_index < ${#AUS_COMBO_LABELS[@]}; combo_index++)); do
    combo_label="${AUS_COMBO_LABELS[$combo_index]}"
    for seed in "${SEEDS[@]}"; do
      python3 "$ADAPTIVE_RUNNER" \
        run-aus-seed \
        --system-root "$SYSTEM_ROOT" \
        --combo-label "$combo_label" \
        --seed "$seed" \
        --bin "$BIN"
      python3 "$REDUCER" \
        process-aus-seed \
        --system-root "$SYSTEM_ROOT" \
        --combo-label "$combo_label" \
        --seed "$seed"
    done
  done
fi

if [[ "$RUN_MTD" != "0" ]]; then
  for ((combo_index = 0; combo_index < ${#MTD_COMBO_LABELS[@]}; combo_index++)); do
    combo_label="${MTD_COMBO_LABELS[$combo_index]}"
    biasfactor="${MTD_COMBO_BIASFACTOR[$combo_index]}"
    for seed in "${SEEDS[@]}"; do
      raw_dir="$SYSTEM_ROOT/MTD/$combo_label/raw/seed_${seed}"
      run_mtd_walker "$raw_dir/left" "$LEFT_X,0.0" "$((seed + combo_index * 10000 + 5000))" "$biasfactor"
      run_mtd_walker "$raw_dir/right" "$RIGHT_X,0.0" "$((seed + combo_index * 10000 + 6000))" "$biasfactor"
      process_args=(process-mtd-seed --system-root "$SYSTEM_ROOT" --combo-label "$combo_label" --seed "$seed")
      if [[ "$seed" == "${SEEDS[0]}" ]]; then
        process_args+=(--make-gif)
      fi
      python3 "$REDUCER" "${process_args[@]}"
    done
  done
fi

if [[ "$RUN_NES" != "0" ]]; then
  for ((combo_index = 0; combo_index < ${#NES_COMBO_LABELS[@]}; combo_index++)); do
    combo_label="${NES_COMBO_LABELS[$combo_index]}"
    k="${NES_COMBO_K[$combo_index]}"
    for seed in "${SEEDS[@]}"; do
      context_dir="$SYSTEM_ROOT/NES/$combo_label/context/seed_${seed}"
      run_nes_eq "$context_dir" "$LEFT_X,0.0" "eq_left.csv" "$((seed + combo_index * 10000 + 11))" "$k"
      run_nes_eq "$context_dir" "$RIGHT_X,0.0" "eq_right.csv" "$((seed + combo_index * 10000 + 29))" "$k"
      for ((time_index = 0; time_index < ${#TIME_STEPS[@]}; time_index++)); do
        total_steps="${TIME_STEPS[$time_index]}"
        label="${TIME_LABELS[$time_index]}"
        t_neq=$((total_steps / (2 * NES_N_TRAJ_PER_DIRECTION)))
        raw_dir="$SYSTEM_ROOT/NES/$combo_label/raw/$label/seed_${seed}"
        run_nes_neq \
          "$raw_dir" \
          "$context_dir/eq_left.csv" \
          "$context_dir/eq_right.csv" \
          "$((seed + combo_index * 10000 + 1000 + time_index))" \
          "$k" \
          "$t_neq"
        process_args=(
          process-nes-seed-time
          --system-root "$SYSTEM_ROOT"
          --combo-label "$combo_label"
          --seed "$seed"
          --time-steps "$total_steps"
        )
        if [[ "$seed" == "${SEEDS[0]}" ]]; then
          process_args+=(--make-gif)
        fi
        if [[ "$time_index" -eq "$((${#TIME_STEPS[@]} - 1))" ]]; then
          process_args+=(--retain-reduced)
        fi
        python3 "$REDUCER" "${process_args[@]}"
      done
      rm -rf "$context_dir"
    done
  done
fi

if [[ "$RUN_MINES" != "0" ]]; then
  for ((combo_index = 0; combo_index < ${#MINES_COMBO_LABELS[@]}; combo_index++)); do
    combo_label="${MINES_COMBO_LABELS[$combo_index]}"
    for seed in "${SEEDS[@]}"; do
      python3 "$ADAPTIVE_RUNNER" \
        run-mines-seed \
        --system-root "$SYSTEM_ROOT" \
        --combo-label "$combo_label" \
        --seed "$seed" \
        --bin "$BIN"
      python3 "$REDUCER" \
        process-mines-seed \
        --system-root "$SYSTEM_ROOT" \
        --combo-label "$combo_label" \
        --seed "$seed"
    done
  done
fi

python3 "$REDUCER" finalize --system-root "$SYSTEM_ROOT"
if [[ "$RUN_PLOTS" != "0" ]]; then
  python3 "$PLOTTER" --system-root "$SYSTEM_ROOT"
fi
if [[ "$RUN_NOTEBOOK" != "0" ]]; then
  BENCHMARK_SYSTEM_ROOT="$SYSTEM_ROOT" jupyter nbconvert --to notebook --execute --inplace "$NOTEBOOK"
fi

echo "Wrote analyzed 1D DoubleWell benchmark data under $SYSTEM_ROOT"
