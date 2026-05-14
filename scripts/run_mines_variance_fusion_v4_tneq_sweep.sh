#!/usr/bin/env bash
set -euo pipefail

SYSTEM_ROOT="${SYSTEM_ROOT:-data/1D/DoubleWell__k0_1p0__x0_m10p0__k1_1p0__x1_10p0__E1_10p0__kT_1p0__dt_0p0005__gamma_1p0}"
BIN="${BIN:-simulations/cpp/neq_sim}"
SEED="${SEED:-123}"
NEQ_NOUT="${NEQ_NOUT:-102}"
NEQ_K_MID_SCALE="${NEQ_K_MID_SCALE:-1.0}"
LOG_DIR="${LOG_DIR:-${SYSTEM_ROOT}/MINES/logs}"
TOTAL_BUDGET_STEPS="${TOTAL_BUDGET_STEPS:-5000000}"
MAX_GENERATIONS="${MAX_GENERATIONS:-10}"
MAX_REFINEMENT_ROUNDS="${MAX_REFINEMENT_ROUNDS:-10}"
LABEL_PREFIX="${LABEL_PREFIX:-mines_variance_fusion_v4}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "scripts/mines_variance_fusion_v4.py" ]]; then
  echo "[MiNES v4 sweep] ERROR: scripts/mines_variance_fusion_v4.py not found" >&2
  exit 1
fi

if [[ ! -f "${SYSTEM_ROOT}/run_context.json" ]]; then
  echo "[MiNES v4 sweep] ERROR: ${SYSTEM_ROOT}/run_context.json not found" >&2
  exit 1
fi

if [[ ! -x "${BIN}" ]]; then
  echo "[MiNES v4 sweep] ERROR: BIN is not executable: ${BIN}" >&2
  exit 1
fi

echo "[MiNES v4 sweep] SYSTEM_ROOT=${SYSTEM_ROOT}"
echo "[MiNES v4 sweep] BIN=${BIN}"
echo "[MiNES v4 sweep] SEED=${SEED}"
echo "[MiNES v4 sweep] LABEL_PREFIX=${LABEL_PREFIX}"
echo "[MiNES v4 sweep] TOTAL_BUDGET_STEPS=${TOTAL_BUDGET_STEPS}"
echo "[MiNES v4 sweep] MAX_GENERATIONS=${MAX_GENERATIONS}"
echo "[MiNES v4 sweep] MAX_REFINEMENT_ROUNDS=${MAX_REFINEMENT_ROUNDS}"
echo "[MiNES v4 sweep] NEQ_NOUT=${NEQ_NOUT}"
echo "[MiNES v4 sweep] NEQ_K_MID_SCALE=${NEQ_K_MID_SCALE}"
echo "[MiNES v4 sweep] LOG_DIR=${LOG_DIR}"

run_mines () {
local t_neq="$1"
  LABEL="${LABEL_PREFIX}_t${t_neq}"
  LOG_FILE="${LOG_DIR}/${LABEL}.log"

  echo "[MiNES v4] Running t_neq=${t_neq}, neq_nout=${NEQ_NOUT}, neq_k_mid_scale=${NEQ_K_MID_SCALE}, label=${LABEL}"
  python scripts/mines_variance_fusion_v4.py \
    --system-root "${SYSTEM_ROOT}" \
    --bin "${BIN}" \
    --seed "${SEED}" \
    --label "${LABEL}" \
    --total-budget-steps "${TOTAL_BUDGET_STEPS}" \
    --max-generations "${MAX_GENERATIONS}" \
    --max-refinement-rounds "${MAX_REFINEMENT_ROUNDS}" \
    --t-neq "${t_neq}" \
    --neq-nout "${NEQ_NOUT}" \
    --neq-k-mid-scale "${NEQ_K_MID_SCALE}" \
    --pmf-method hybrid \
    --target-kl 5 \
    --eq-overlap-threshold 0.3 \
    --final-refinement-mode eq-extend \
    --target-mbar-ddf 1e-3 \
    > "${LOG_FILE}" 2>&1

  echo "[MiNES v4] Finished t_neq=${t_neq}; log=${LOG_FILE}"

}

for T_NEQ in 1000 2000 3000 5000; do
  run_mines "${T_NEQ}"
done
