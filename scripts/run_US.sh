#!/usr/bin/env bash
# Run only the Umbrella Sampling (US) benchmark arm.
# All parameters are forwarded to run_US_MTD_NES.sh.
# Accepts the same environment variable overrides (POT_*, US_K_VALUES_CSV, SEEDS_CSV, etc.).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

RUN_US=1 RUN_AUS=0 RUN_NES=0 RUN_MINES=0 RUN_MTD=0 RUN_PLOTS=0 RUN_NOTEBOOK=0 \
  bash "$SCRIPT_DIR/run_US_MTD_NES.sh" "$@"
