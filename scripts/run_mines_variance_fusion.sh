#!/usr/bin/env bash
set -euo pipefail

SYSTEM_ROOT="data/1D/DoubleWell__k0_1p0__x0_m10p0__k1_1p0__x1_10p0__E1_10p0__kT_1p0__dt_0p0005__gamma_1p0"
BIN="simulations/cpp/neq_sim"
SEED=10101
LABEL="mines_variance_fusion"
TOTAL_BUDGET_STEPS=5000000
T_NEQ=2000
N_NEQ_TRAJ=30
N_EQ_STEPS=10000
JS_THRESHOLD=0.3
NEQ_CONNECTIVITY_THRESHOLD=0.3
ALLOW_PARTIAL_NEQ_BUDGET=0
MIN_NEQ_TRAJ=20
MAX_RESCUE_ROUNDS=10
ANALYSIS_XMIN=""
ANALYSIS_XMAX=""
QUICK_TEST=0
S_RESCUE=2.0
RESCUE_CENTER_F_SLOPE=0.5
RESCUE_CENTER_F_START=2.0
RESCUE_CENTER_F_MIN=-2.0
RESCUE_CENTER_F_MAX=1.0
NEQ_PROTOCOL_MODE="GT"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system-root)
      SYSTEM_ROOT="$2"
      shift 2
      ;;
    --bin)
      BIN="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --total-budget-steps)
      TOTAL_BUDGET_STEPS="$2"
      shift 2
      ;;
    --t-neq)
      T_NEQ="$2"
      shift 2
      ;;
    --n-neq-traj)
      N_NEQ_TRAJ="$2"
      shift 2
      ;;
    --n-eq-steps)
      N_EQ_STEPS="$2"
      shift 2
      ;;
    --js-threshold)
      JS_THRESHOLD="$2"
      shift 2
      ;;
    --neq-connectivity-threshold)
      NEQ_CONNECTIVITY_THRESHOLD="$2"
      shift 2
      ;;
    --allow-partial-neq-budget)
      ALLOW_PARTIAL_NEQ_BUDGET=1
      shift
      ;;
    --min-neq-traj)
      MIN_NEQ_TRAJ="$2"
      shift 2
      ;;
    --analysis-xmin)
      ANALYSIS_XMIN="$2"
      shift 2
      ;;
    --analysis-xmax)
      ANALYSIS_XMAX="$2"
      shift 2
      ;;
    --quick-test)
      QUICK_TEST=1
      shift
      ;;
    --s-rescue)
      S_RESCUE="$2"
      shift 2
      ;;
    --rescue-center-f-slope)
      RESCUE_CENTER_F_SLOPE="$2"
      shift 2
      ;;
    --rescue-center-f-start)
      RESCUE_CENTER_F_START="$2"
      shift 2
      ;;
    --rescue-center-f-min)
      RESCUE_CENTER_F_MIN="$2"
      shift 2
      ;;
    --rescue-center-f-max)
      RESCUE_CENTER_F_MAX="$2"
      shift 2
      ;;
    --neq-protocol-mode)
      NEQ_PROTOCOL_MODE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

PYTHON_SCRIPT="scripts/mines_variance_fusion.py"
RUN_CONTEXT="${SYSTEM_ROOT}/run_context.json"
LOG_DIR="${SYSTEM_ROOT}/MINES/${LABEL}/logs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Missing Python runner: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${RUN_CONTEXT}" ]]; then
  echo "Missing run_context.json: ${RUN_CONTEXT}" >&2
  exit 1
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Simulation binary is missing or not executable: ${BIN}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

CMD=(
  python "${PYTHON_SCRIPT}"
  --system-root "${SYSTEM_ROOT}"
  --bin "${BIN}"
  --seed "${SEED}"
  --label "${LABEL}"
  --total-budget-steps "${TOTAL_BUDGET_STEPS}"
  --t-neq "${T_NEQ}"
  --n-neq-traj "${N_NEQ_TRAJ}"
  --n-eq-steps "${N_EQ_STEPS}"
  --js-threshold "${JS_THRESHOLD}"
  --neq-connectivity-threshold "${NEQ_CONNECTIVITY_THRESHOLD}"
  --min-neq-traj "${MIN_NEQ_TRAJ}"
  --s-rescue "${S_RESCUE}"
  --rescue-center-f-slope "${RESCUE_CENTER_F_SLOPE}"
  --rescue-center-f-start "${RESCUE_CENTER_F_START}"
  --rescue-center-f-min "${RESCUE_CENTER_F_MIN}"
  --rescue-center-f-max "${RESCUE_CENTER_F_MAX}"
  --neq-protocol-mode "${NEQ_PROTOCOL_MODE}"
  --max-rescue-rounds "${MAX_RESCUE_ROUNDS}" # Limit rescue rounds to at most the number of NEQ trajectories
)

if [[ "${QUICK_TEST}" -eq 1 ]]; then
  CMD+=(--quick-test)
fi

if [[ "${ALLOW_PARTIAL_NEQ_BUDGET}" -eq 1 ]]; then
  CMD+=(--allow-partial-neq-budget)
fi

if [[ -n "${ANALYSIS_XMIN}" ]]; then
  CMD+=(--analysis-xmin "${ANALYSIS_XMIN}")
fi

if [[ -n "${ANALYSIS_XMAX}" ]]; then
  CMD+=(--analysis-xmax "${ANALYSIS_XMAX}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
