#!/usr/bin/env bash
set -euo pipefail

# Run the active MiNES current-protocol simulation until the protocol stops on
# its own.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

DEFAULT_SYSTEM_ROOT="$ROOT_DIR/data/1D/DoubleWell__k0_1p0__x0_m10p0__k1_1p0__x1_10p0__E1_10p0__kT_1p0__dt_0p0005__gamma_1p0"
SYSTEM_ROOT="${MINES_SYSTEM_ROOT:-$DEFAULT_SYSTEM_ROOT}"
LABEL="${MINES_LABEL:-current_protocol_t5000_n50}"
SEED="${MINES_SEED:-101}"
BIN="${MINES_BIN:-$ROOT_DIR/simulations/cpp/neq_sim}"
T_NEQ="${MINES_T_NEQ:-5000}"
N_NEQ_TRAJ="${MINES_N_NEQ_TRAJ:-50}"
TOTAL_BUDGET_STEPS="${MINES_TOTAL_BUDGET_STEPS:-200000}"

if [[ ! -x "$BIN" ]]; then
  echo "Expected MiNES binary at $BIN" >&2
  exit 1
fi

printf 'Running MiNES current protocol: system_root=%s label=%s seed=%s t_neq=%s n_neq_traj=%s total_budget_steps=%s\n' \
  "$SYSTEM_ROOT" "$LABEL" "$SEED" "$T_NEQ" "$N_NEQ_TRAJ" "$TOTAL_BUDGET_STEPS"

python "$ROOT_DIR/simulations/adaptive_methods.py" run-mines-current-protocol \
  --system-root "$SYSTEM_ROOT" \
  --seed "$SEED" \
  --bin "$BIN" \
  --label "$LABEL" \
  --t-neq "$T_NEQ" \
  --n-neq-traj "$N_NEQ_TRAJ" \
  --total-budget-steps "$TOTAL_BUDGET_STEPS"

printf 'Raw MiNES outputs live under %s/MINES/%s/raw/seed_%s\n' "$SYSTEM_ROOT" "$LABEL" "$SEED"
