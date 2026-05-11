#!/usr/bin/env bash
set -euo pipefail

# Run the active 1D benchmark workflow for three requested direct-sampling
# DoubleWell systems. This delegates to the single-system runner, so it covers
# the current screened `US`, `AUS`, `NES`, `MINES`, and `WT-MTD` methods.
# 1. k0 = k1 = 1.0, E1 = 10
# 2. k0 = k1 = 0.5, E1 = 10
# 3. k0 = k1 = 1.0, E1 = 1
#
# The shared settings remain:
# - x0 = -10
# - x1 = 10
# - thermal_kT = 1.0
# - dt = 0.0005
# - gamma = 1.0
#
# By default this script skips notebook execution because the notebook kernel
# startup may need permissions outside the sandbox.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER="$ROOT_DIR/simulations/run_US_MTD_NES.sh"

RUN_US="${RUN_US:-1}"
RUN_AUS="${RUN_AUS:-1}"
RUN_NES="${RUN_NES:-1}"
RUN_MINES="${RUN_MINES:-1}"
RUN_MTD="${RUN_MTD:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"
RUN_NOTEBOOK="${RUN_NOTEBOOK:-0}"

X0="${X0:--10.0}"
X1="${X1:-10.0}"
THERMAL_KT="${THERMAL_KT:-1.0}"
DT="${DT:-0.0005}"
GAMMA="${GAMMA:-1.0}"

run_case() {
  local pot_k0="$1"
  local pot_k1="$2"
  local pot_e1="$3"

  POT_K0="$pot_k0" \
  POT_K1="$pot_k1" \
  POT_E1="$pot_e1" \
  POT_X0="$X0" \
  POT_X1="$X1" \
  THERMAL_KT="$THERMAL_KT" \
  DT="$DT" \
  GAMMA="$GAMMA" \
  RUN_US="$RUN_US" \
  RUN_AUS="$RUN_AUS" \
  RUN_NES="$RUN_NES" \
  RUN_MINES="$RUN_MINES" \
  RUN_MTD="$RUN_MTD" \
  RUN_PLOTS="$RUN_PLOTS" \
  RUN_NOTEBOOK="$RUN_NOTEBOOK" \
  bash "$RUNNER"
}

run_case 1.0 1.0 10.0
run_case 0.5 0.5 10.0
run_case 1.0 1.0 1.0
